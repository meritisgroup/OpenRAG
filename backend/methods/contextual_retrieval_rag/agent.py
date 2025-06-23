from ...base_classes import RagAgent
from ...utils.agent import get_Agent
from ...utils.factory_vectorbase import get_vectorbase
from ...database.database_class import get_database
from .prompts import prompts
from .indexation import ContextualRetrievalIndexation
from ..naive_rag.query import NaiveSearch
import numpy as np
from ..query_reformulation.query_reformulation import query_reformulation


class ContextualRetrievalRagAgent(RagAgent):
    """
    For each chunk an LLM is asked to read the whole document and to generate in a short paragraph / sentence explaining the chunk given the context.
    """

    def __init__(
        self,
        config_server: dict,
        db_name: str = "db_contextual_retrieval_rag",
        vb_name: str = "vb_contextual_retrieval_rag",
    ) -> None:
        """
        Args:
            model (str): model used to generate context, to be set in backend/methods/contextual_retrieval_rag/config.json file
            storage_path: folder in which database will be stored
            params_host_llm(dict): parameters for Ollama or VLLM, to be set in backend/config_server.json file
            params_vectorbase(dict): vectorbase connection parameters, to be set in backend/config_server.json file
            embedding_model (str): Model used to embed documents and queries, to be set in backend/methods/contextual_retrieval_rag/config.json file
            language (str) : Sets the language of the prompts (available "EN", "FR"), to be set in in backend/methods/contextual_retrieval_rag/config.json file
            api_key (str) : API key to be used if needed, to be set in backend/config_server.json file (not mandatory if using Ollama or VLLM)
            db_name (str) : Name given to the database that keeps track of already processed docs, if it already exists adds new documents to the existing database (stored in storage/ folder)
            vb_name (str) : Name given to the vectorbase, if it already exists adds new documents to the existing vectorbase (stored in milvus/elasticsearch docker)
        """
        self.storage_path = config_server["storage_path"]
        self.language = config_server["language"]
        self.type_text_splitter = config_server["TextSplitter"]
        self.type_retrieval = config_server["type_retrieval"]
        self.embedding_model = config_server["embedding_model"]
        self.db_name = vb_name
        self.vb_name = db_name
        self.nb_chunks = config_server["nb_chunks"]
        self.agent = get_Agent(config_server)
        self.db = get_database(db_name=self.db_name, storage_path=self.storage_path)
        self.vb = get_vectorbase(
            vb_name=self.vb_name,
            config_server=config_server,
            agent=self.agent,
        )
        self.prompts = prompts[self.language]
        self.reformulate_query = config_server["reformulate_query"]
        if self.reformulate_query:
            self.reformulater = query_reformulation(
                agent=self.agent, language=self.language
            )

    def get_nb_token_embeddings(self):
        return self.vb.get_nb_token_embeddings()

    def indexation_phase(
        self,
        path_input: str,
        reset_index: bool = False,
        chunk_size: int = 500,
        overlap: bool = True,
    ) -> None:
        """
        Does the indexation of a given knowledge base, full process is located in indexation.py
        Args:
            path_input (str) : where the documents to be processed are stored
            chunk_size (str) : number of characters in each chunk
            overlap (bool) : Wether chunks overlap each other
        """
        if reset_index:
            self.vb.delete_collection()
            self.db.clean_database()

        index = ContextualRetrievalIndexation(
            data_path=path_input,
            db=self.db,
            vb=self.vb,
            language=self.language,
            agent=self.agent,
            type_text_splitter=self.type_text_splitter,
            embedding_model=self.embedding_model,
        )
        index.run_pipeline(batch=True)

        return None

    def get_rag_context(
        self, query: str, nb_chunks: int = 5
    ) -> tuple:
        """
        Takes a query and retrieves a given number of chunks using the NaiveSearch implemented in backend/methods/naive_rag/query.py

        Args:
            query (str) : The query that needs answering
            nb_chunks (int) : Number of chunks to be retrieved

        Output:
            context (str) : All retrieved chunks, seperated by a new line and '[...]'
            name_docs : Source document name for each retrieved chunk
        """
        ns = NaiveSearch(vector_base=self.vb, nb_chunks=nb_chunks)
        context, name_docs = ns.get_context(query=query)
        return context, name_docs

    def get_rag_contexts(
        self, queries: list[str], nb_chunks: int = 5
    ):
        contexts = []
        names_docs = []
        for query in queries:
            context, name_docs = self.get_rag_context(
                query=query, nb_chunks=nb_chunks
            )
            contexts.append(context)
            names_docs.append(name_docs)
        return contexts, names_docs

    def generate_answers(self, queries: list[str], nb_chunks: int = 2):
        answers = []
        for query in queries:
            answer = self.generate_answer(query=query, nb_chunks=nb_chunks)
            answers.append(answer)
        return answers

    def generate_answer(self, query: str, nb_chunks: int = 2):
        """
        Takes a query, retrieves appropriated context and generates an answer
        Args:
            query (str) : The query that needs answering
            method_parameter (int): Number of chunks to be retrieved

        Output:
            answer (str) : The answer to the query given the retrieved context
        """
        input_tokens = 0
        output_tokens = 0
        impact = [0, 0, ""]
        energy = [0, 0, ""]
        if self.reformulate_query:
            query, input_tokens, output_tokens, impact, energy = (
                self.reformulater.reformulate(query=query,
                                              nb_reformulation=1)
            )
            query = query[0]
        context, _ = self.get_rag_context(query=query, nb_chunks=nb_chunks)
        prompt = self.prompts["smooth_generation"]["QUERY_TEMPLATE"].format(
            context=context, query=query
        )

        system_prompt = self.prompts["smooth_generation"]["SYSTEM_PROMPT"]

        answer = self.agent.predict(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        nb_input_tokens = np.sum(answer["nb_input_tokens"]) + input_tokens
        nb_output_tokens = np.sum(answer["nb_output_tokens"]) + output_tokens

        impacts = answer["impacts"]
        impacts[0] += impact[0]
        impacts[1] += impact[1]

        energies = answer["energy"]
        energies[0] += energy[0]
        energies[1] += energy[1]

        return {
            "answer": answer["texts"],
            "nb_input_tokens": nb_input_tokens,
            "nb_output_tokens": nb_output_tokens,
            "context": context,
            "impacts": impacts,
            "energy": energies,
        }

    def release_gpu_memory(self):
        self.agent.release_memory()
