import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from database.rag_classes import Document
from database.data_extraction import DocumentText
from utils.splitter import TextSplitter
from .wiki_structure import PageType, slugify, parse_wikilinks
from .wiki_manager import WikiManager
from .prompts import PROMPTS

logger = logging.getLogger(__name__)


class LlmWikiIndexation:
    def __init__(
        self,
        wiki_manager: WikiManager,
        data_manager,
        agent,
        llm_model: str,
        splitter,
        language: str = "EN",
        storage_path: str = "./storage",
    ):
        self.wiki_manager = wiki_manager
        self.data_manager = data_manager
        self.agent = agent
        self.llm_model = llm_model
        self.language = language
        self.storage_path = storage_path
        self.splitter = splitter
        self.prompts = PROMPTS[self.language]

    def _check_new_documents(self) -> tuple:
        docs_already_processed = []
        for doc in self.data_manager.query(Document):
            if doc.path:
                docs_already_processed.append(Path(doc.path).resolve().as_posix())

        all_doc_paths = []
        for db_name in self.data_manager.get_dbs_name():
            all_doc_paths.extend(
                self.data_manager.get_list_path_documents(db_name=db_name)
            )

        to_process_norm = [Path(p).resolve().as_posix() for p in all_doc_paths]
        docs_to_process = [
            doc for doc in to_process_norm if doc not in docs_already_processed
        ]
        return (len(docs_to_process) > 0, docs_to_process)

    def _parse_document(self, path: str) -> str:
        doc_text = DocumentText(
            doc_index=0,
            path=path,
            config_server={"data_preprocessing": "raw"},
            agent=self.agent,
            splitter=self.splitter,
            reset_preprocess=False,
        )
        return doc_text.get_content()

    def _analyze_source(self, content: str, source_name: str) -> tuple[dict, int, int]:
        schema = self.wiki_manager.read_schema()
        existing_index = self.wiki_manager.read_index()

        prompt_template = self.prompts["analyze_source"]["QUERY_TEMPLATE"]
        system_prompt = self.prompts["analyze_source"]["SYSTEM_PROMPT"]

        truncated = content[:15000] if len(content) > 15000 else content
        prompt = prompt_template.format(
            schema=schema,
            existing_index=existing_index,
            source_name=source_name,
            content=truncated,
        )

        result = self.agent.predict(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.llm_model,
        )

        analysis = self._parse_json_response(result["texts"], {})
        in_t = result.get("nb_input_tokens", 0)
        out_t = result.get("nb_output_tokens", 0)
        return analysis, in_t, out_t

    def _detect_contradictions(self, analysis: dict) -> list:
        contradictions = analysis.get("contradictions_with_wiki", [])
        if not contradictions:
            return []

        pages_to_update = analysis.get("pages_to_update", [])
        for c in contradictions:
            wiki_page_title = c.get("wiki_page", "")
            slug = slugify(wiki_page_title)
            for pt in [PageType.ENTITY, PageType.CONCEPT]:
                page = self.wiki_manager.read_page(pt, slug)
                if page:
                    pages_to_update.append(
                        {
                            "title": wiki_page_title,
                            "page_type": pt.value,
                            "new_facts": [c.get("new_info", "")],
                        }
                    )
                    break

        analysis["pages_to_update"] = pages_to_update

        return contradictions

    def _generate_pages(
        self, analysis: dict, contradictions: list, source_name: str
    ) -> tuple[list, int, int]:
        schema = self.wiki_manager.read_schema()

        analysis_json = json.dumps(analysis, ensure_ascii=False, indent=2)
        contradictions_json = json.dumps(contradictions, ensure_ascii=False, indent=2)

        existing_pages_text = self._build_existing_pages_text(analysis)

        prompt_template = self.prompts["generate_pages"]["QUERY_TEMPLATE"]
        system_prompt = self.prompts["generate_pages"]["SYSTEM_PROMPT"]

        prompt = prompt_template.format(
            schema=schema,
            source_name=source_name,
            analysis=analysis_json,
            contradictions=contradictions_json,
            existing_pages=existing_pages_text,
        )

        result = self.agent.predict(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.llm_model,
        )

        pages = self._parse_json_response(result["texts"], [])
        in_t = result.get("nb_input_tokens", 0)
        out_t = result.get("nb_output_tokens", 0)
        return pages, in_t, out_t

    def _build_existing_pages_text(self, analysis: dict) -> str:
        pages_to_update = analysis.get("pages_to_update", [])
        if not pages_to_update:
            return "No existing pages to update."

        parts = []
        for p in pages_to_update:
            title = p.get("title", "")
            page_type_str = p.get("page_type", "entity")
            try:
                pt = PageType(page_type_str)
            except ValueError:
                pt = PageType.ENTITY
            slug = slugify(title)
            existing = self.wiki_manager.read_page(pt, slug)
            if existing:
                parts.append(
                    f"--- Page: {pt.value}/{slug} ---\nTitle: {existing.title}\nContent:\n{existing.content}\n"
                )

        return "\n".join(parts) if parts else "No existing pages found for update."

    def _save_pages(self, pages: list, source_name: str) -> tuple[list, list]:
        created = []
        updated = []
        today = date.today().isoformat()

        for page_data in pages:
            page_type_str = page_data.get("page_type", "entity")
            try:
                pt = PageType(page_type_str)
            except ValueError:
                pt = PageType.ENTITY

            title = page_data.get("title", "Untitled")
            slug = page_data.get("slug", slugify(title))
            content = page_data.get("content", "")
            frontmatter = page_data.get("frontmatter", {})

            if "sources" not in frontmatter:
                frontmatter["sources"] = [source_name]
            elif source_name not in frontmatter["sources"]:
                frontmatter["sources"].append(source_name)

            frontmatter["title"] = title
            frontmatter["type"] = pt.value
            if "created" not in frontmatter:
                frontmatter["created"] = today
            frontmatter["updated"] = today

            action = page_data.get("action", "create")
            is_update = self.wiki_manager.page_exists(pt, slug)

            self.wiki_manager.write_page(pt, slug, content, frontmatter)

            page_id = f"{pt.value}/{slug}"
            if is_update:
                updated.append(page_id)
            else:
                created.append(page_id)

        return created, updated

    def _save_document_record(
        self,
        source_name: str,
        path_doc: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        for db_name in self.data_manager.get_dbs_name():
            doc_record = Document(
                name=source_name,
                path=path_doc,
                embedding_tokens=0,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            self.data_manager.add_instance(instance=doc_record, db_name=db_name)

    def _finalize_pipeline(self, all_details: list) -> None:
        self.wiki_manager.update_index()

        details_text = "\n".join(all_details)
        self.wiki_manager.append_log("ingest", details_text)

    def _update_overview_if_needed(self) -> tuple[int, int]:
        pages = self.wiki_manager.list_pages()
        if len(pages) >= 3:
            all_pages_text = self._build_all_pages_summary(pages)
            prompt_template = self.prompts["update_overview"]["QUERY_TEMPLATE"]
            system_prompt = self.prompts["update_overview"]["SYSTEM_PROMPT"]
            prompt = prompt_template.format(all_pages=all_pages_text)
            result = self.agent.predict(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.llm_model,
            )
            self.wiki_manager.update_overview(result["texts"])
            return result.get("nb_input_tokens", 0), result.get("nb_output_tokens", 0)
        return 0, 0

    def _build_all_pages_summary(self, pages: list) -> str:
        parts = []
        for page in pages:
            parts.append(
                f"## {page.title} ({page.page_type.value})\n{page.content[:500]}\n"
            )
        return "\n".join(parts)

    def _parse_json_response(self, text: str, default):
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
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return default

    def run_pipeline(
        self,
        config_server: dict,
        reset_preprocess: bool = False,
        progress_callback=None,
    ) -> None:
        self.wiki_manager.initialize()

        need_extraction, docs_to_process = self._check_new_documents()

        if not need_extraction:
            if progress_callback:
                progress_callback(1.0, "No new documents to process")
            return

        total_docs = len(docs_to_process)
        total_input_tokens = 0
        total_output_tokens = 0
        all_details = []

        for idx, path_doc in enumerate(docs_to_process):
            source_name = Path(path_doc).name
            progress = idx / total_docs

            try:
                if progress_callback:
                    progress_callback(progress, f"Analyzing {source_name}...")

                content = self._parse_document(path_doc)
                if not content:
                    logger.warning(f"Empty content for {source_name}, skipping")
                    continue

                doc_input_tokens = 0
                doc_output_tokens = 0

                analysis, in_t, out_t = self._analyze_source(content, source_name)
                doc_input_tokens += in_t
                doc_output_tokens += out_t

                contradictions = self._detect_contradictions(analysis)

                if progress_callback:
                    progress_callback(
                        progress + 0.3 / total_docs,
                        f"Generating pages for {source_name}...",
                    )

                pages, in_t, out_t = self._generate_pages(analysis, contradictions, source_name)
                doc_input_tokens += in_t
                doc_output_tokens += out_t

                if not pages:
                    pages = self._fallback_page_generation(analysis, source_name)

                if progress_callback:
                    progress_callback(
                        progress + 0.6 / total_docs, f"Saving pages for {source_name}..."
                    )

                created, updated = self._save_pages(pages, source_name)

                details = f"source: {source_name}"
                if created:
                    details += f"\nPages creees: {len(created)} ({', '.join(created[:5])}{'...' if len(created) > 5 else ''})"
                if updated:
                    details += f"\nPages mises a jour: {len(updated)} ({', '.join(updated[:5])}{'...' if len(updated) > 5 else ''})"
                all_details.append(details)

                self._save_document_record(
                    source_name, path_doc, doc_input_tokens, doc_output_tokens
                )

                total_input_tokens += doc_input_tokens
                total_output_tokens += doc_output_tokens

            except Exception as e:
                logger.error(f"Failed to process {source_name}: {e}")
                continue

        if all_details:
            self._finalize_pipeline(all_details)

        if progress_callback:
            progress_callback(0.9, "Updating overview...")

        in_t, out_t = self._update_overview_if_needed()
        total_input_tokens += in_t
        total_output_tokens += out_t

        if progress_callback:
            progress_callback(1.0, "Indexation completed")

    def _fallback_page_generation(self, analysis: dict, source_name: str) -> list:
        today = date.today().isoformat()
        pages = []
        slug = slugify(source_name)

        summary = analysis.get("source_summary", "No summary available.")
        pages.append(
            {
                "action": "create",
                "page_type": "source",
                "title": source_name,
                "slug": slug,
                "frontmatter": {
                    "title": source_name,
                    "type": "source",
                    "summary": summary[:120],
                    "sources": [source_name],
                    "tags": ["auto-generated"],
                },
                "content": f"# {source_name}\n\n{summary}\n",
            }
        )

        for entity in analysis.get("entities", []):
            entity_slug = slugify(entity.get("name", ""))
            if not entity_slug:
                continue
            facts = "\n".join(f"- {f}" for f in entity.get("facts", []))
            desc = entity.get("description", "")
            pages.append(
                {
                    "action": "create",
                    "page_type": "entity",
                    "title": entity.get("name", ""),
                    "slug": entity_slug,
                    "frontmatter": {
                        "title": entity.get("name", ""),
                        "type": "entity",
                        "summary": desc[:120] if desc else f"Entity: {entity.get('name', '')}",
                        "sources": [source_name],
                        "tags": [entity.get("type", "other")],
                    },
                    "content": f"# {entity.get('name', '')}\n\n{desc}\n\n## Key Facts\n{facts}\n",
                }
            )

        for concept in analysis.get("concepts", []):
            concept_slug = slugify(concept.get("name", ""))
            if not concept_slug:
                continue
            facts = "\n".join(f"- {f}" for f in concept.get("facts", []))
            desc = concept.get("description", "")
            pages.append(
                {
                    "action": "create",
                    "page_type": "concept",
                    "title": concept.get("name", ""),
                    "slug": concept_slug,
                    "frontmatter": {
                        "title": concept.get("name", ""),
                        "type": "concept",
                        "summary": desc[:120] if desc else f"Concept: {concept.get('name', '')}",
                        "sources": [source_name],
                        "tags": ["auto-generated"],
                    },
                    "content": f"# {concept.get('name', '')}\n\n{desc}\n\n## Key Facts\n{facts}\n",
                }
            )

        return pages
