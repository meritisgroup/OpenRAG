from ...utils.factory_vectorbase import get_vectorbase
from ...utils.agent_functions import get_system_prompt
from ..advanced_rag.agent import AdvancedRag
from .indexation import QbRagIndexation
from .query import QbSearch
from ...utils.agent import get_Agent
from ...database.rag_classes import Question, Document, Chunk
from .prompts import prompts
import numpy as np
from ..query_reformulation.query_reformulation import query_reformulation
from ...database.database_class import get_management_data


class QueryBasedRagAgent(AdvancedRag):
    """
    RAG Agent based on Query-Based principle meaning we pass to a LLM all our chunked documents for asking it to generate questions about chunks.
    Then we associate the generated questions with their related chunks and when a query match a question we give as context the relevant chunk.
    """

    def __init__(
        self,
        config_server: dict,
        dbs_name: list[str],
        data_folders_name: list[str]
    ) -> None:
        """ """
        super().__init__(config_server=config_server,
                        dbs_name = dbs_name,
                        data_folders_name = data_folders_name)
        
        self.prompts = prompts[self.language]
        self.system_prompt = get_system_prompt(config_server, self.prompts)
        self.data_manager = get_management_data(dbs_name=self.dbs_name,
                                                data_folders_name=self.data_folders_name,
                                                storage_path=self.storage_path,
                                                config_server=config_server,
                                                agent=self.agent)
        self.data_manager.add_table(Question)
        self.data_manager.add_table(Chunk)

    def indexation_phase(
        self,
        reset_index: bool = False,
        reset_preprocess: bool = False,
        overlap: bool = True
    ) -> None:
        """Indexation phase for QB Rag"""

        if reset_preprocess:
            reset_index = True
        if reset_index:
            self.data_manager.delete_collection()
            self.data_manager.clean_database()

        qb_index = QbRagIndexation(
            language=self.language,
            agent=self.agent,
            data_manager=self.data_manager,
            type_text_splitter=self.type_text_splitter,
            embedding_model=self.embedding_model,
        )

        qb_index.run_pipeline(chunk_size=self.chunk_size,
                              config_server=self.config_server,
                              chunk_overlap=overlap,
                              reset_preprocess=reset_preprocess)

    def get_rag_context(
        self,
        query: str,
        nb_chunks: int = 5,
        model: str = None,
        to_prompt = False
    ) -> str:
        """ """
        agent = self.agent

        qs = QbSearch(
            agent=agent,
            data_manager=self.data_manager,
            nb_questions=nb_chunks,
            language=self.language,
        )

        context, docs_name = qs.get_context(query=query,
                                            to_prompt=to_prompt)

        return context, docs_name

    def release_gpu_memory(self):
        self.agent.release_memory()

    def get_rag_contexts(self, queries: list[str], nb_chunks: int = 5):
        contexts = []
        names_docs = []
        for query in queries:
            context, name_docs = self.get_rag_context(query=query, nb_chunks=nb_chunks)
            contexts.append(context)
            names_docs.append(name_docs)
        return contexts, names_docs

    def generate_answers(self, queries: list[str], nb_chunks: int = 2, options_generation = None):
        answers = []
        for query in queries:
            answer = self.generate_answer(query=query,
                                          nb_chunks=nb_chunks,
                                          options_generation=options_generation)
            answers.append(answer)
        return answers
