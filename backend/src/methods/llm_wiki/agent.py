import json
import logging
import shutil
from datetime import date
from pathlib import Path

from application.agents.base_rag_agent import BaseRAGAgent
from core.error_handler import handle_errors, LLMError
from database.rag_classes import Document, Tokens
from utils.agent_functions import get_system_prompt
from utils.splitter import get_splitter
from .wiki_manager import WikiManager
from .wiki_structure import PageType, slugify
from .indexation import LlmWikiIndexation
from .query import LlmWikiSearch
from .lint import WikiLinter
from .prompts import PROMPTS

logger = logging.getLogger(__name__)


class LlmWikiAgent(BaseRAGAgent):
    def __init__(
        self,
        config_server: dict,
        models_infos: dict,
        dbs_name: list,
        data_folders_name: list,
    ) -> None:
        super().__init__(
            config_server=config_server,
            models_infos=models_infos,
            dbs_name=dbs_name,
            data_folders_name=data_folders_name,
            rag_name="llm_wiki",
        )
        self.data_manager.add_table(Document)
        self.data_manager.add_table(Tokens)
        self.prompts = PROMPTS[self.language]

        wiki_path = config_server.get("wiki_path") or str(
            Path(self.storage_path) / "wiki"
        )
        self.wiki_manager = WikiManager(
            wiki_path=Path(wiki_path), language=self.language
        )

        self.splitter = get_splitter(
            type_text_splitter=self.type_text_splitter,
            data_preprocessing=config_server.get("data_preprocessing", "raw"),
            agent=self.agent,
            embedding_model=self.embedding_model,
        )

    def indexation_phase(
        self,
        reset_index: bool = False,
        reset_preprocess: bool = False,
        overlap: bool = True,
        progress_callback=None,
        **kwargs,
    ) -> None:
        if reset_preprocess:
            reset_index = True

        if reset_index:
            if self.wiki_manager.wiki_path.exists():
                shutil.rmtree(self.wiki_manager.wiki_path)
            for db_name in self.dbs_name:
                self.data_manager.clean_database(db_name=db_name)

        indexation = LlmWikiIndexation(
            wiki_manager=self.wiki_manager,
            data_manager=self.data_manager,
            agent=self.agent,
            llm_model=self.llm_model,
            splitter=self.splitter,
            language=self.language,
            storage_path=self.storage_path,
        )
        indexation.run_pipeline(
            config_server=self.config_server,
            reset_preprocess=reset_preprocess,
            progress_callback=progress_callback,
        )

    def get_rag_context(self, query: str, nb_chunks: int = 10):
        search = LlmWikiSearch(
            agent=self.agent,
            llm_model=self.llm_model,
            wiki_manager=self.wiki_manager,
            language=self.language,
            nb_chunks=nb_chunks,
        )
        return search.get_context(query=query)

    @handle_errors(reraise=True, exception_types=(LLMError,))
    def generate_answer(
        self, query: str, nb_chunks: int = 10, options_generation=None
    ) -> dict:
        (impacts, energies) = ([0, 0, ""], [0, 0, ""])

        if self.reformulate_query:
            (query, input_t, output_t, impacts, energies) = (
                self._reformulate_query_if_needed(query=query, nb_reformulation=1)
            )

        (context, chunks, tokens_counter) = self.get_rag_context(
            query=query, nb_chunks=nb_chunks
        )
        self.add_tokens(
            tokens_counter["nb_input_tokens"], tokens_counter["nb_output_tokens"]
        )

        prompt_template = self.prompts["synthesize_answer"]["QUERY_TEMPLATE"]
        system_prompt = self.prompts["synthesize_answer"]["SYSTEM_PROMPT"]
        schema = self.wiki_manager.read_schema()
        pages_text = context if context else ""
        prompt = prompt_template.format(pages=pages_text, query=query, schema=schema)

        if options_generation is None:
            options_generation = self.config_server.get("options_generation", {})

        answer = self.agent.predict(
            prompt=prompt,
            system_prompt=system_prompt,
            options_generation=options_generation,
            model=self.llm_model,
        )
        self.aggregate_response_tokens(answer)

        impacts[2] = answer["impacts"][2]
        impacts[0] += answer["impacts"][0]
        impacts[1] += answer["impacts"][1]
        energies[2] = answer["energy"][2]
        energies[0] += answer["energy"][0]
        energies[1] += answer["energy"][1]

        self._maybe_crystallize(query, answer["texts"])

        return self._build_response(
            answer_text=answer["texts"],
            context=chunks,
            query=query,
            impacts=impacts,
            energy=energies,
        )

    def _maybe_crystallize(self, query: str, answer_text: str) -> None:
        if not self.config_server.get("crystallize", False):
            return

        prompt_template = self.prompts["crystallize"]["QUERY_TEMPLATE"]
        system_prompt = self.prompts["crystallize"]["SYSTEM_PROMPT"]

        truncated_answer = answer_text[:5000] if len(answer_text) > 5000 else answer_text
        prompt = prompt_template.format(query=query, answer=truncated_answer)

        try:
            result = self.agent.predict(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.llm_model,
            )
            self.aggregate_response_tokens(result)

            eval_data = self._parse_json_response(result["texts"], {})
            if not eval_data or not eval_data.get("should_crystallize"):
                return

            page_content = eval_data.get("page_content", "")
            if not page_content:
                return

            title = eval_data.get("suggested_title", query[:80])
            summary = eval_data.get("suggested_summary", "")
            tags = eval_data.get("suggested_tags", [])
            slug = slugify(title)
            today = date.today().isoformat()

            frontmatter = {
                "title": title,
                "type": "query",
                "sources": [],
                "tags": tags + ["crystallized"],
                "created": today,
                "updated": today,
                "original_query": query,
            }
            if summary:
                frontmatter["summary"] = summary

            self.wiki_manager.write_page(PageType.QUERY, slug, page_content, frontmatter)
            self.wiki_manager.update_index()
            self.wiki_manager.append_log("crystallize", f"Query: {query[:100]} -> {title}")

            logger.info(f"Crystallized Q&A as wiki page: queries/{slug}")

        except Exception as e:
            logger.warning(f"Crystallization failed (non-critical): {e}")

    @staticmethod
    def _parse_json_response(text: str, default):
        try:
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                json_str = text[start:end]
            elif "[" in text:
                start = text.index("[")
                end = text.rindex("]") + 1
                json_str = text[start:end]
            else:
                return default
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return default

    def lint(self, fix: bool = False):
        linter = WikiLinter(
            wiki_manager=self.wiki_manager,
            agent=self.agent,
            llm_model=self.llm_model,
            language=self.language,
        )
        return linter.lint(fix=fix)
