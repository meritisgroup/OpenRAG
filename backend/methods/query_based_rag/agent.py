from ...utils.factory_vectorbase import get_vectorbase
from ...utils.agent_functions import get_system_prompt
from ..naive_rag.agent import NaiveRagAgent
from .indexation import QbRagIndexation
from .query import QbSearch
from ...utils.agent import get_Agent
from ...database.rag_classes import Question, Document, Chunk
from .prompts import prompts
import numpy as np
from ..query_reformulation.query_reformulation import query_reformulation
from ...database.database_class import get_management_data


class QueryBasedRagAgent(NaiveRagAgent):
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
        self.dbs_name = dbs_name
        self.data_folders_name = data_folders_name
        self.embedding_model = config_server["embedding_model"]

        self.language = config_server["language"]
        self.prompts = prompts[self.language]
        self.system_prompt = get_system_prompt(config_server, self.prompts)

        self.chunk_size = config_server["chunk_length"]

        self.type_text_splitter = config_server["TextSplitter"]
        self.type_retrieval = config_server["type_retrieval"]
        self.nb_chunks = config_server["nb_chunks"]
        self.storage_path = config_server["storage_path"]
        self.config_server = config_server
        self.agent = get_Agent(config_server)
        self.data_manager = get_management_data(dbs_name=self.dbs_name,
                                                data_folders_name=self.data_folders_name,
                                                storage_path=self.storage_path,
                                                config_server=config_server,
                                                agent=self.agent)
        self.data_manager.add_table(Question)
        self.data_manager.add_table(Chunk)

        self.params_vectorbase = config_server["params_vectorbase"]
        self.reformulate_query = config_server["reformulate_query"]
        if self.reformulate_query:
            self.reformulater = query_reformulation(
                agent=self.agent, language=self.language
            )

    def indexation_phase(
        self,
        reset_index: bool = False,
        overlap: bool = True,
    ) -> None:
        """Indexation phase for QB Rag"""
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
                              chunk_overlap=overlap,)

    def get_rag_context(
        self,
        query: str,
        method_parameter: int = 2,
        model: str = None,
    ) -> str:
        """ """
        agent = self.agent

        qs = QbSearch(
            agent=agent,
            data_manager=self.data_manager,
            nb_questions=method_parameter,
            language=self.language,
        )

        context, docs_name = qs.get_context(query=query)

        return context, docs_name

    def generate_answer(
        self,
        query: str,
        nb_chunks: str = 2,
        model: str = None,
        options_generation = None
    ) -> str:
        """ """
        agent = self.agent
        nb_input_tokens = 0
        nb_output_tokens = 0
        impacts, energies = [0, 0, ""], [0, 0, ""]
        if self.reformulate_query:
            query, input_t, output_t, impacts, energies = self.reformulater.reformulate(
                query=query, nb_reformulation=1
            )
            query = query[0]
            nb_input_tokens += input_t
            nb_output_tokens = output_t

        context, docs_name = self.get_rag_context(
            query=query, method_parameter=nb_chunks, model=model
        )
        prompt = self.prompts["smooth_generation"]["QUERY_TEMPLATE"].format(
            context=context, query=query
        )
        if options_generation is None:
            options_generation = self.config_server["options_generation"]

        answer = agent.predict(prompt=prompt, system_prompt=self.system_prompt,
                               options_generation=options_generation)
        nb_input_tokens += np.sum(answer["nb_input_tokens"])
        nb_output_tokens += np.sum(answer["nb_output_tokens"])

        impacts[2] = answer["impacts"][2]
        impacts[0] += answer["impacts"][0]
        impacts[1] += answer["impacts"][1]
        energies[2] = answer["energy"][2]
        energies[0] += answer["energy"][0]
        energies[1] += answer["energy"][1]

        return {
            "answer": answer["texts"],
            "nb_input_tokens": nb_input_tokens,
            "nb_output_tokens": nb_output_tokens,
            "docs_name": docs_name,
            "context": context,
            "impacts": impacts,
            "energy": energies,
        }

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
