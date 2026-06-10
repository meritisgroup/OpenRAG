import json
import logging
from typing import Optional

from base_classes import Search
from database.rag_classes import Chunk
from .wiki_manager import WikiManager
from .wiki_structure import PageType
from .prompts import PROMPTS

logger = logging.getLogger(__name__)


class LlmWikiSearch(Search):
    def __init__(
        self,
        agent,
        llm_model: str,
        wiki_manager: WikiManager,
        language: str = "EN",
        nb_chunks: int = 10,
    ):
        super().__init__(agent)
        self.llm_model = llm_model
        self.wiki_manager = wiki_manager
        self.language = language
        self.nb_chunks = nb_chunks
        self.prompts = PROMPTS[self.language]

    def get_context(self, query: str):
        tokens_counter = {"nb_input_tokens": 0, "nb_output_tokens": 0}

        index_text = self.wiki_manager.read_index()
        if not index_text or "No pages yet" in index_text:
            return ("", [], tokens_counter)

        selected_paths = self._navigate_index(query, index_text, tokens_counter)
        if not selected_paths:
            return ("", [], tokens_counter)

        pages_content = []
        chunks = []
        for i, path_str in enumerate(selected_paths[: self.nb_chunks]):
            parts = path_str.split("/")
            if len(parts) != 2:
                continue
            pt_str, slug = parts
            try:
                pt = PageType(pt_str)
            except ValueError:
                continue

            page = self.wiki_manager.read_page(pt, slug)
            if page:
                content = f"[{i + 1}] **{page.title}** ({page.page_type.value})\n{page.content}"
                pages_content.append(content)
                chunks.append(
                    Chunk(
                        text=page.content,
                        document=page.title,
                        position_in_doc=i + 1,
                        id=f"wiki_{pt_str}_{slug}",
                    )
                )

        if not pages_content:
            return ("", [], tokens_counter)

        context_str = "\n\n---\n\n".join(pages_content)
        return (context_str, chunks, tokens_counter)

    def _navigate_index(
        self, query: str, index_text: str, tokens_counter: dict
    ) -> list:
        prompt_template = self.prompts["navigate_index"]["QUERY_TEMPLATE"]
        system_prompt = self.prompts["navigate_index"]["SYSTEM_PROMPT"]

        schema = self.wiki_manager.read_schema()
        prompt = prompt_template.format(query=query, index=index_text, schema=schema)

        result = self.agent.predict(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.llm_model,
        )
        tokens_counter["nb_input_tokens"] += result.get("nb_input_tokens", 0)
        tokens_counter["nb_output_tokens"] += result.get("nb_output_tokens", 0)

        return self._parse_selection(result["texts"])

    def _parse_selection(self, llm_output: str) -> list:
        try:
            json_str = llm_output
            if "{" in llm_output:
                start = llm_output.index("{")
                end = llm_output.rindex("}") + 1
                json_str = llm_output[start:end]
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "selected" in parsed:
                return [str(s).strip() for s in parsed["selected"] if s]
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM navigation output: {e}")
        return []
