import logging
import json
from typing import Optional
from database.rag_classes import Chunk, Section
from base_classes import Search
from .prompts import PROMPTS

logger = logging.getLogger(__name__)


class ReasoningSearch(Search):

    def __init__(self, agent, model, data_manager, language='EN', nb_chunks=10, max_depth=2):
        super().__init__(agent)
        self.llm_model = model
        self.data_manager = data_manager
        self.language = language
        self.nb_chunks = nb_chunks
        self.max_depth = max_depth
        self.prompts = PROMPTS[self.language]

    def get_context(self, query: str):
        tokens_counter = {'nb_input_tokens': 0, 'nb_output_tokens': 0}

        all_sections = list(self.data_manager.query(Section))
        if not all_sections:
            return ('', [], tokens_counter)

        documents = [s for s in all_sections if s.level == 0]
        if not documents:
            sections = [s for s in all_sections if s.level >= 1]
            if not sections:
                return ('', [], tokens_counter)
            context_str = '\n\n'.join(s.text for s in sections[:self.nb_chunks])
            chunks = [Chunk(text=s.text, document=s.document, position_in_doc=s.position, id=s.id) for s in sections[:self.nb_chunks]]
            return (context_str, chunks, tokens_counter)

        # Step 1: LLM navigation at document level
        selected_docs = self._navigate_documents(query, documents, tokens_counter)
        if not selected_docs:
            selected_docs = documents[:3]

        # Step 2: LLM navigation at section level
        doc_names = {d.title for d in selected_docs}
        sections_level_1 = [s for s in all_sections if s.level == 1 and s.document in doc_names]

        selected_sections = self._navigate_sections(query, sections_level_1, tokens_counter)
        if not selected_sections:
            selected_sections = sections_level_1[:5]

        # Step 3: Collect paragraphs under selected sections
        selected_section_ids = {s.id for s in selected_sections}
        paragraphs = [s for s in all_sections if s.level == 2 and s.parent_id in selected_section_ids]

        if not paragraphs:
            result_sections = selected_sections
        else:
            result_sections = paragraphs

        result_sections = result_sections[:self.nb_chunks]

        context_parts = []
        chunks = []
        for section in result_sections:
            header = f"[{section.document}"
            if section.level > 0:
                header += f" - {section.title}"
            header += "]"
            context_parts.append(f"{header}\n{section.text}")
            chunks.append(Chunk(text=section.text, document=section.document, position_in_doc=section.position, id=section.id))

        context_str = '\n\n---\n\n'.join(context_parts)
        return (context_str, chunks, tokens_counter)

    def _navigate_documents(self, query: str, documents: list, tokens_counter: dict) -> list:
        summaries = []
        for i, doc in enumerate(documents):
            summary_text = doc.summary if doc.summary else doc.text[:500]
            summaries.append(f"{i + 1}. **{doc.title}**: {summary_text}")

        summaries_text = '\n'.join(summaries)
        prompt = self.prompts['navigate_documents']['QUERY_TEMPLATE'].format(query=query, summaries=summaries_text)
        system_prompt = self.prompts['navigate_documents']['SYSTEM_PROMPT']

        result = self.agent.predict(prompt=prompt, system_prompt=system_prompt, model=self.llm_model)
        tokens_counter['nb_input_tokens'] += result.get('nb_input_tokens', 0)
        tokens_counter['nb_output_tokens'] += result.get('nb_output_tokens', 0)

        selected_titles = self._parse_selection(result['texts'])
        selected = [d for d in documents if d.title in selected_titles]

        if not selected:
            selected = documents[:3]
        return selected

    def _navigate_sections(self, query: str, sections: list, tokens_counter: dict) -> list:
        if not sections:
            return []

        summaries = []
        for i, section in enumerate(sections):
            summary_text = section.summary if section.summary else section.text[:500]
            summaries.append(f"{i + 1}. **{section.title}**: {summary_text}")

        summaries_text = '\n'.join(summaries)
        prompt = self.prompts['navigate_sections']['QUERY_TEMPLATE'].format(query=query, summaries=summaries_text)
        system_prompt = self.prompts['navigate_sections']['SYSTEM_PROMPT']

        result = self.agent.predict(prompt=prompt, system_prompt=system_prompt, model=self.llm_model)
        tokens_counter['nb_input_tokens'] += result.get('nb_input_tokens', 0)
        tokens_counter['nb_output_tokens'] += result.get('nb_output_tokens', 0)

        selected_titles = self._parse_selection(result['texts'])
        selected = [s for s in sections if s.title in selected_titles]
        return selected

    def _parse_selection(self, llm_output: str) -> list[str]:
        try:
            json_str = llm_output
            if '{' in llm_output:
                start = llm_output.index('{')
                end = llm_output.rindex('}') + 1
                json_str = llm_output[start:end]
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and 'selected' in parsed:
                return [str(s).strip() for s in parsed['selected'] if s]
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM navigation output: {e}")
        return []
