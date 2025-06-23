from ..advanced_rag.agent import AdvancedRag
from ..advanced_rag.query import NaiveSearch
import numpy as np
from itertools import chain


class RerankerRag(AdvancedRag):
    """
    A Rag framework that reranks naively retrieved documents using a reranker model. Only available using VLLM
    """

    def __init__(
        self,
        config_server: dict,
        db_name: str = "db_naive_rag",
        vb_name: str = "vb_naive_rag",
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
            db_name=db_name,
            vb_name=vb_name,
        )
        self.nb_chunks = config_server["nb_chunks"]

    def get_rag_context(
        self, query: str, nb_chunks: int = 5
    ) -> list[str]:
        """
        Takes a query and retrieves a given number of chunks using the NaiveSearch implemented in backend/methods/naive_rag/query.py

        Args:
            query (str) : The query that needs answering
            nb_chunks (int) : Number of chunks to be retrieved

        Output:
            context (str) : All retrieved chunks, seperated by a new line and '[...]'
        """

        ns = NaiveSearch(vector_base=self.vb, nb_chunks=nb_chunks)
        context = ns.get_context(query=query)
        return context

    def get_nb_token_embeddings(self):
        return self.vb.get_nb_token_embeddings()

    def contexts_to_prompts(self, contexts):
        """
        Takes a list of retrieved chunks and formats them into a single char
        Args:
            contexts (list[str]) : List of all retrieved chunks

        Returns:
            context (str): Concatenated chunks separated by new lines and '[...]'
        """
        context = ""
        for chunk in contexts:
            if chunk not in context:
                context += chunk + "\n[...]\n"
        return context

    def release_gpu_memory(self):
        self.agent.release_memory()

    def generate_answer(
        self,
        query: str,
        nb_chunks: int = 7,
        nb_reformulation=5,
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
        impacts, energies = [0, 0, ''], [0, 0, '']
        if self.reformulate_query:
            queries, input_t, output_t, immpacts, energies = self.reformulater.reformulate(
                query=query, nb_reformulation=nb_reformulation
            )
            nb_input_tokens += input_t
            nb_output_tokens += output_t
        else:
            queries = [query]

        contexts = [
            self.get_rag_context(query=query, nb_chunks=nb_chunks)
            for query in queries
        ]
        contexts = list(chain(*contexts))
        contexts = list(set(contexts))
        if len(contexts) > 0 and self.rerank:
            contexts, _, input_tokens = self.reranker.rerank(
                query=query, contexts=contexts, max_contexts=20
            )
            nb_input_tokens += input_tokens

        context = self.contexts_to_prompts(contexts=contexts)
        prompt = self.prompts["smooth_generation"]["QUERY_TEMPLATE"].format(
            context=context, query=query
        )
        system_prompt = self.prompts["smooth_generation"]["SYSTEM_PROMPT"]
        answer = agent.predict(prompt=prompt, system_prompt=system_prompt)
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
            "impacts" : impacts,
            "energy" : energies
        }

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
