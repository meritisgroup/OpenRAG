# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:37:47 2025

@author: chardy
"""
from .indexation import NaiveRagIndexation
from .query import NaiveSearch
from ..naive_rag.agent import NaiveRagAgent
from .reranker import Reranker
from ..query_reformulation.query_reformulation import query_reformulation
import numpy as np
from itertools import chain


class AdvancedRag(NaiveRagAgent):
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
            config_server : RAG configuration
            db_name: Name given to database that will store indexed documents' name
            vb_name: Name given to vectorbase that will store chunks embeddings

        """

        super().__init__(config_server=config_server, db_name=db_name, vb_name=vb_name)

        self.reranker_model = config_server["reranker_model"]
        self.reformulate_query = config_server["reformulate_query"]
        self.language = config_server["language"]
        self.type_processor_chunks = type_processor_chunks
        self.nb_chunks = config_server["nb_chunks"]
        self.nb_input_tokens = 0
        self.nb_output_tokens = 0
        if self.reranker_model is not None:
            self.reranker = Reranker(
                agent=self.agent, reranking_model=self.reranker_model
            )
            self.rerank = True
        else:
            self.rerank = False

        if self.reformulate_query:
            self.reformulater = query_reformulation(
                agent=self.agent, language=self.language
            )

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

        index = NaiveRagIndexation(
                                data_path=path_input,
                                db=self.db,
                                vb=self.vb,
                                type_text_splitter=self.type_text_splitter,
                                agent=self.agent,
                                embedding_model=self.embedding_model,
                                type_processor_chunks=self.type_processor_chunks,
                                language=self.language,
                            )

        index.run_pipeline(
                        chunk_size=chunk_size,
                        chunk_overlap=overlap,
                        batch=self.params_vectorbase["batch"],
                    )

        return None

    def get_rag_context(
        self, query: str, nb_chunks: int = 5) -> list[str]:
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

        contexts = [
            self.get_rag_context(query=query, nb_chunks=nb_chunks)
            for query in queries
        ]
        contexts = list(chain(*contexts))
        contexts = list(set(contexts))

        if len(contexts) > 0 and self.rerank:
            contexts, _, nb_input_tokens = self.reranker.rerank(
                query=query, contexts=contexts, max_contexts=20
            )
            self.nb_input_tokens += nb_input_tokens

        context = self.contexts_to_prompts(contexts=contexts)

        prompt = self.prompts["smooth_generation"]["QUERY_TEMPLATE"].format(
            context=context, query=query
        )
        system_prompt = self.prompts["smooth_generation"]["SYSTEM_PROMPT"]
        answer = agent.predict(prompt=prompt, system_prompt=system_prompt)
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
            "context": context,
            "impacts": impact,
            "energy": energies,
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
