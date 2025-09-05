from ..advanced_rag.agent import AdvancedRag
from ..naive_rag.agent import NaiveRagAgent
from ..advanced_rag.query import NaiveSearch
from ..naive_rag.indexation import contexts_to_prompts
import numpy as np
from itertools import chain
from backend.database.rag_classes import Chunk


class RerankerRag(AdvancedRag):
    """
    A Rag framework that reranks naively retrieved documents using a reranker model. Only available using VLLM
    """

    def __init__(
        self,
        config_server: dict,
        dbs_name: list[str],
        data_folders_name: list[str],
        type_processor_chunks: list[str] = [],
    ) -> None:
        """
        Args:
            model (str): model used to generate context, to be set in backend/methods/advanced_rag_reranker/config.json file
            storage_path: folder in which database will be stored
            params_host_llm(dict): parameters for Ollama or VLLM, to be set in backend/config_server.json file
            params_vectorbase(dict): vectorbase connection parameters, to be set in backend/config_server.json file
            embedding_model (str): Model used to embed documents and queries, to be set in backend/methods/advanced_rag_reranker/config.json file
            reranker_model (str) : Model used to rerank documents, to be set in backend/methods/advanced_rag_reranker/config.json fil
            language (str) : Sets the language of the prompts (available "EN", "FR"), to be set in in backend/methods/advanced_rag_reranker/config.json file
            api_key (str) : API key to be used if needed, to be set in backend/config_server.json file (not mandatory if using Ollama or VLLM)
            db_name (str) : Name given to the database that keeps track of already processed docs, if it already exists adds new documents to the existing database (stored in storage/ folder)
            vb_name (str) : Name given to the vectorbase, if it already exists adds new documents to the existing vectorbase (stored in milvus/elasticsearch docker)
            type_retrieval (str) : How documents will be retrieved (embeddings, BM25, vlm_embeddings are available plus hybrid if using elasticsearch)
        """
        config_server["reformulate_query"] = False
        super().__init__(
            config_server=config_server,
            dbs_name=dbs_name,
            data_folders_name=data_folders_name,
        )
        self.nb_chunks = config_server["nb_chunks"]

    def get_rag_context(self, query: str, nb_chunks: int = 5) -> list[list[Chunk]]:
        """
        Takes a query and retrieves a given number of chunks using the NaiveSearch implemented

        Args:
            query (str) : The query that needs answering
            nb_chunks (int) : Number of chunks to be retrieved
        Output:
            context (list[str]) : All retrieved chunks
        """
        ns = NaiveSearch(data_manager=self.data_manager, nb_chunks=nb_chunks)
        chunk_lists = ns.get_context(query=query)
        return chunk_lists

    def release_gpu_memory(self):
        self.agent.release_memory()

    def generate_answer(
        self,
        query: str,
        nb_chunks: int = 7,
        nb_reformulation=5,
        options_generation=None,
    ) -> str:
        """
        Takes a query, retrieves appropriated context and generates an answer
        Args:
            query (str) : The query that needs answering
            nb_chunks (int): Number of chunks to be retrieved
            nb_reformulation (int): How many reformulation of the query will be computed and fed to an LLM
        Output:
            answer (str) : The answer to the query given the retrieved context
        """
        agent = self.agent
        impacts = [0, 0, ""]
        energy = [0, 0, ""]
        if self.reformulate_query:
            queries, input_t, output_t, impacts, energy = self.reformulater.reformulate(
                query=query, nb_reformulation=nb_reformulation
            )
            self.nb_input_tokens += input_t
            self.nb_output_tokens += output_t
        else:
            queries = [query]

        # Building the prompt in several steps
        chunk_lists = self.get_rag_context(query=queries[0], nb_chunks=nb_chunks)

        chunk_list = [chunk for chunk_list in chunk_lists for chunk in chunk_list]
        docs_name = [chunk.document for chunk in chunk_list]

        if len(chunk_list) > 0 and self.rerank:
            rerank_chunk_list, additional_data, nb_input_tokens = self.reranker.rerank(
                query=query,
                chunk_list=chunk_list,
                max_contexts=len(chunk_list),
                additional_data={"docs_name": docs_name},
            )
            self.nb_input_tokens += nb_input_tokens
        else:
            rerank_chunk_list = chunk_list

        prompt = self.build_final_prompt([rerank_chunk_list], query)

        if options_generation is None:
            options_generation = self.config_server["options_generation"]

        answer = agent.predict(
            prompt=prompt,
            system_prompt=self.system_prompt,
            options_generation=options_generation,
        )
        self.nb_input_tokens += np.sum(answer["nb_input_tokens"])
        self.nb_output_tokens += np.sum(answer["nb_output_tokens"])

        impact = answer["impacts"]
        impact[0] += impacts[0]
        impact[1] += impacts[1]

        energies = answer["energy"]
        energies[0] += energy[0]
        energies[1] += energy[1]

        return {
            "answer": answer["texts"],
            "nb_input_tokens": self.nb_input_tokens,
            "nb_output_tokens": self.nb_output_tokens,
            "context": rerank_chunk_list,
            "impacts": impact,
            "energy": energies,
        }

    # def get_rag_contexts(self, queries: list[str], nb_chunks: int = 5):
    #     contexts = []
    #     names_docs = []
    #     for query in queries:
    #         context, name_docs = self.get_rag_context(query=query, nb_chunks=nb_chunks)
    #         contexts.append(context)
    #         names_docs.append(name_docs)
    #     return contexts, names_docs

    def generate_answers(
        self, queries: list[str], nb_chunks: int = 2, options_generation=None
    ):
        answers = []
        for query in queries:
            answer = self.generate_answer(
                query=query, nb_chunks=nb_chunks, options_generation=options_generation
            )
            answers.append(answer)
        return answers
