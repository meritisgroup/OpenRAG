import logging
import json
from pathlib import Path
from typing import Optional
import numpy as np
import concurrent.futures

from database.rag_classes import Document, Section, Chunk, Tokens
from database.data_extraction import DocumentText
from utils.splitter import get_splitter
from utils.progress import ProgressBar, TwoLevelProgressTracker
from utils.threading_utils import get_executor_threads
from .hierarchical_tree import SectionNode, build_tree_from_chunks, build_document_node
from .prompts import PROMPTS

logger = logging.getLogger(__name__)


class ReasoningRagIndexation:

    def __init__(self, data_manager, storage_path: str, agent, type_text_splitter: str, data_preprocessing: str, embedding_model: str, llm_model: str, language: str = 'EN'):
        self.data_manager = data_manager
        self.agent = agent
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.language = language
        self.storage_path = storage_path
        self.prompts = PROMPTS[self.language]
        self.splitter = get_splitter(type_text_splitter=type_text_splitter, data_preprocessing=data_preprocessing, agent=self.agent, embedding_model=embedding_model)

    def _check_extraction_necessity(self, db_name: str = None):
        docs_already_processed = [res[0] for res in self.data_manager.query(Document.path)]
        to_process_norm = [Path(p).resolve().as_posix() for p in self.data_manager.get_list_path_documents(db_name=db_name)]
        docs_already_norm = [Path(p).resolve().as_posix() for p in docs_already_processed]
        docs_to_process = [doc for doc in to_process_norm if doc not in docs_already_norm]
        return (len(docs_to_process) > 0, docs_to_process)

    def _parse_document(self, path_doc: str, chunk_size: int, chunk_overlap: bool) -> list[SectionNode]:
        doc_text = DocumentText(
            doc_index=0,
            path=path_doc,
            config_server={'data_preprocessing': 'raw'},
            agent=self.agent,
            splitter=self.splitter,
            reset_preprocess=False,
        )
        content = doc_text.get_content()
        if not content:
            return []
        chunks = self.splitter.split_text(text=content, chunk_size=chunk_size, overlap=chunk_overlap)
        doc_name = Path(path_doc).name
        return build_tree_from_chunks(chunks, document_name=doc_name)

    def _generate_summaries(self, nodes: list[SectionNode], tracker=None) -> list[SectionNode]:
        nodes_to_summarize = [n for n in nodes if not n.summary and n.level < 2]
        if not nodes_to_summarize:
            return nodes

        prompt_template = self.prompts['summarize_section']['QUERY_TEMPLATE']
        system_prompt = self.prompts['summarize_section']['SYSTEM_PROMPT']

        prompts = []
        system_prompts = []
        for node in nodes_to_summarize:
            prompts.append(prompt_template.format(title=node.title, content=node.text[:3000]))
            system_prompts.append(system_prompt)

        results = self.agent.multiple_predict(
            prompts=prompts,
            system_prompts=system_prompts,
            model=self.llm_model,
        )

        for i, node in enumerate(nodes_to_summarize):
            if i < len(results['texts']):
                node.summary = results['texts'][i]

        return nodes

    def _save_sections_to_db(self, nodes: list[SectionNode], db_name: str = None):
        all_nodes = []
        for node in nodes:
            all_nodes.extend(node.flatten())

        for node in all_nodes:
            section = Section(
                id=node.id,
                parent_id=node.parent_id or '',
                document=node.document,
                level=node.level,
                title=node.title,
                summary=node.summary,
                text=node.text,
                position=node.position,
            )
            self.data_manager.add_instance(instance=section, db_name=db_name)

    def _embed_sections(self, nodes: list[SectionNode], db_name: str, tracker=None):
        all_nodes = []
        for node in nodes:
            all_nodes.extend(node.flatten())

        chunks_to_embed = []
        for node in all_nodes:
            text_to_embed = node.summary if node.summary else node.text
            chunks_to_embed.append(Chunk(
                text=text_to_embed,
                document=node.document,
                position_in_doc=node.position,
                id=node.id,
            ))

        batch_size = 500
        for i in range(0, len(chunks_to_embed), batch_size):
            batch = chunks_to_embed[i:i + batch_size]
            path_docs = [None] * len(batch)
            self.data_manager.add_str_batch_elements(
                chunks=batch,
                path_docs=path_docs,
                display_message=False,
                collection_name='reasoning_rag',
                vb_name=db_name,
            )
            if tracker:
                tracker.increment_sub(f"Embedded {min(i + batch_size, len(chunks_to_embed))}/{len(chunks_to_embed)} sections")

    def run_pipeline(self, config_server: dict, chunk_size: int = 1024, chunk_overlap: bool = True, reset_preprocess: bool = False, progress_callback=None):
        dbs_name = self.data_manager.get_dbs_name()
        total_steps = len(dbs_name) * 4
        tracker = None
        if progress_callback:
            tracker = TwoLevelProgressTracker(total_steps, progress_callback)
            tracker.update_global("Starting Reasoning RAG indexation", step_name="Initializing")

        for db_name in dbs_name:
            (need_extraction, docs_to_process) = self._check_extraction_necessity(db_name=db_name)

            if need_extraction:
                # Step 1: Parse documents and build tree
                if tracker:
                    tracker.update_global(f"Processing database: {db_name}", step_name="Parsing documents")
                    tracker.set_sub_total(len(docs_to_process))

                all_root_nodes = []
                for idx, path_doc in enumerate(docs_to_process):
                    root_nodes = self._parse_document(path_doc, chunk_size, chunk_overlap)
                    all_root_nodes.extend(root_nodes)
                    if tracker:
                        tracker.increment_sub(f"Parsed {idx + 1}/{len(docs_to_process)} documents")

                # Step 2: Generate summaries
                if tracker:
                    tracker.complete_step("Parsing completed")
                    tracker.update_global(f"Generating summaries", step_name="Summarization")

                all_nodes_flat = []
                for root in all_root_nodes:
                    all_nodes_flat.extend(root.flatten())
                self._generate_summaries(all_nodes_flat, tracker=tracker)

                # Step 3: Save to DB
                if tracker:
                    tracker.complete_step("Summarization completed")
                    tracker.update_global(f"Saving to database", step_name="Database")

                doc_nodes = []
                for path_doc in docs_to_process:
                    doc_name = Path(path_doc).name
                    doc_sections = [n for n in all_root_nodes if n.document == doc_name]
                    if doc_sections:
                        doc_node = build_document_node(doc_name, doc_sections)
                        doc_node.summary = ' '.join(s.summary for s in doc_sections if s.summary)[:1000]
                        doc_nodes.append(doc_node)

                self._generate_summaries(doc_nodes, tracker=None)
                all_nodes_to_save = doc_nodes + all_root_nodes
                self._save_sections_to_db(all_nodes_to_save, db_name=db_name)

                for path_doc in docs_to_process:
                    doc_name = Path(path_doc).name
                    doc_record = Document(
                        name=doc_name,
                        path=str(path_doc),
                        embedding_tokens=0,
                        input_tokens=0,
                        output_tokens=0,
                    )
                    self.data_manager.add_instance(instance=doc_record, db_name=db_name)

                # Step 4: Embed sections
                if tracker:
                    tracker.complete_step("Database save completed")
                    tracker.update_global(f"Embedding sections", step_name="Embeddings")
                    tracker.set_sub_total(len(all_nodes_to_save))

                self.data_manager.create_collection(vb_name=db_name, name='reasoning_rag')
                self._embed_sections(all_nodes_to_save, db_name=db_name, tracker=tracker)

                if tracker:
                    tracker.complete_step("Embeddings completed")
            else:
                if tracker:
                    for _ in range(4):
                        tracker.complete_step("Skipped (already indexed)")

        if tracker:
            tracker.complete_all()
