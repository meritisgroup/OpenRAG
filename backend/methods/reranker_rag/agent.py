from ..advanced_rag.agent import AdvancedRag
from ..advanced_rag.query import NaiveSearch
from ..naive_rag.indexation import contexts_to_prompts
import numpy as np
from itertools import chain


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
            dbs_name = dbs_name,
            data_folders_name = data_folders_name
        )
        self.nb_chunks = config_server["nb_chunks"]

    def get_rag_context(self, query: str, nb_chunks: int = 5) -> list[str]:
        """
        Takes a query and retrieves a given number of chunks using the NaiveSearch implemented in backend/methods/naive_rag/query.py

        Args:
            query (str) : The query that needs answering
            nb_chunks (int) : Number of chunks to be retrieved

        Output:
            context (str) : All retrieved chunks, seperated by a new line and '[...]'
        """

        ns = NaiveSearch(data_manager=self.data_manager, 
                         nb_chunks=nb_chunks)
        context, docs_name = ns.get_context(query=query)
        return context, docs_name

    def release_gpu_memory(self):
        self.agent.release_memory()

    def generate_answer(
        self,
        query: str,
        nb_chunks: int = 7,
        nb_reformulation=5,
        options_generation = None
    ) -> str:
        """
        Takes a query, retrieves appropriated context and generates an answer
        Args:
            query (str) : The query that needs answering
            model (str) :
            nb_chunks (int): Number of chunks to be retrieved
        Output:
            answer (str) : The answer to the query given the retrieved context
        """
        agent = self.agent
        nb_input_tokens = 0
        nb_output_tokens = 0
        impacts, energies = [0, 0, ""], [0, 0, ""]
        if self.reformulate_query:
            queries, input_t, output_t, immpacts, energies = (
                self.reformulater.reformulate(
                    query=query, nb_reformulation=nb_reformulation
                )
            )
            nb_input_tokens += input_t
            nb_output_tokens += output_t
        else:
            queries = [query]


        results = [
            self.get_rag_context(query=query, nb_chunks=nb_chunks) for query in queries
        ]
        contexts = []
        docs_name = []
        for result in results:
            contexts+=result[0]
            docs_name+=result[1]

        if len(contexts) > 0 and self.rerank:
            contexts, additional_data, input_tokens = self.reranker.rerank(
                query=query, contexts=contexts, max_contexts=20,
                additional_data={"docs_name": docs_name})
            nb_input_tokens += input_tokens

        docs_name = additional_data["docs_name"]

        context, docs_name = contexts_to_prompts(contexts=contexts,
                                                 docs_name=docs_name)
        prompt = self.prompts["smooth_generation"]["QUERY_TEMPLATE"].format(
            context=context, query=query
        )

        if options_generation is None:
            options_generation = self.config_server["options_generation"]

        answer = agent.predict(prompt=prompt, 
                               system_prompt=self.system_prompt,
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
            "context": context,
            "docs_name": docs_name,
            "impacts": impacts,
            "energy": energies,
        }

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
