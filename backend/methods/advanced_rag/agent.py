# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:37:47 2025

@author: chardy
"""
from .indexation import AdvancedIndexation
from ..naive_rag.indexation import contexts_to_prompts
from ..naive_rag.agent import NaiveRagAgent
from ..naive_rag.query import NaiveSearch
from .reranker import Reranker
from ..query_reformulation.query_reformulation import query_reformulation
import numpy as np
from itertools import chain
from backend.database.rag_classes import Chunk


class AdvancedRag(NaiveRagAgent):
    """
    A Rag framework that reranks naively retrieved documents using a reranker model. Only available using VLLM
    """

    def __init__(
        self,
        config_server: dict,
        models_infos: dict,
        dbs_name: list[str],
        data_folders_name: list[str]
    ) -> None:
        """
        Args:
            config_server : RAG configuration
            db_name: Name given to database that will store indexed documents' name
            vb_name: Name given to vectorbase that will store chunks embeddings

        """

        super().__init__(
            config_server=config_server,
            models_infos=models_infos,
            dbs_name=dbs_name,
            data_folders_name=data_folders_name)

        self.reranker_model = config_server["reranker_model"]
        self.type_processor_chunks = config_server["ProcessorChunks"]

        if self.reranker_model is not None:
            self.reranker = Reranker(
                agent=self.agent, reranking_model=self.reranker_model
            )
            self.rerank = True
        else:
            self.rerank = False


    def indexation_phase(
        self,
        reset_index: bool = False,
        reset_preprocess: bool = False,
        overlap: bool = True,
    ) -> None:
        """
        Does the indexation of a given knowledge base, full process is located in indexation.py
        Args:
            path_input (str) : where the documents to be processed are stored
            overlap (bool) : Wether chunks overlap each other

        """
        if reset_preprocess:
            reset_index = True
            
        if reset_index:
            self.data_manager.delete_collection()
            self.data_manager.clean_database()

        index = AdvancedIndexation(
            data_manager=self.data_manager,
            type_text_splitter=self.type_text_splitter,
            data_preprocessing=self.config_server["data_preprocessing"],
            agent=self.agent,
            embedding_model=self.embedding_model,
            llm_model=self.llm_model,
            type_processor_chunks=self.type_processor_chunks,
            language=self.language,
        )

        index.run_pipeline(
            chunk_size=self.chunk_size,
            chunk_overlap=overlap,
            config_server=self.config_server,
            reset_preprocess=reset_preprocess,
            max_workers=self.config_server["max_workers"]
        )

        return None

    def get_rag_context(self, query: str, nb_chunks: int = 5) -> list[list[Chunk]]:
        """
        Takes a query and retrieves a given number of chunks using the NaiveSearch implemented

        Args:
            query (str) : The query that needs answering
            nb_chunks (int) : Number of chunks to be retrieved
        Output:
            context (list[str]) : All retrieved chunks
        """
        ns = NaiveSearch(data_manager=self.data_manager, 
                         nb_chunks=nb_chunks)
        chunk_lists = ns.get_context(query=query)
        return chunk_lists

    def get_nb_token_embeddings(self):
        return self.data_manager.get_nb_token_embeddings()

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
        nb_input_tokens = 0
        nb_output_tokens = 0
        if self.reformulate_query:
            queries, input_t, output_t, impacts, energy = self.reformulater.reformulate(
                query=query, nb_reformulation=nb_reformulation
            )
            nb_input_tokens += np.sum(input_t)
            nb_output_tokens += np.sum(output_t)
        else:
            queries = [query]

        chunk_lists = self.get_rag_context(query=queries[0], 
                                           nb_chunks=nb_chunks)
        chunk_list = [chunk for chunk_list in chunk_lists for chunk in chunk_list]

        if len(chunk_list) > 0 and self.rerank:
            rerank_chunk_list, additional_data, nb_input_tokens_rerank = self.reranker.rerank(
                                                                    query=query,
                                                                    chunk_list=chunk_list,
                                                                    max_contexts=self.config_server["nb_chunks_reranker"],
                                                                )
            #nb_input_tokens += np.sum(nb_input_tokens_rerank)
        else:
            rerank_chunk_list = chunk_list

        prompt = self.build_final_prompt(rerank_chunk_list, query)

        if options_generation is None:
            options_generation = self.config_server["options_generation"]

        answer = agent.predict(prompt=prompt,
                               system_prompt=self.system_prompt,
                               options_generation=options_generation,
                               model=self.llm_model)
        
        nb_input_tokens += np.sum(answer["nb_input_tokens"])
        nb_output_tokens += np.sum(answer["nb_output_tokens"])

        impact = answer["impacts"]
        impact[0] += impacts[0]
        impact[1] += impacts[1]

        energies = answer["energy"]
        energies[0] += energy[0]
        energies[1] += energy[1]

        return {
            "answer": answer["texts"],
            "nb_input_tokens": nb_input_tokens,
            "nb_output_tokens": nb_output_tokens,
            "context": rerank_chunk_list,
            "impacts": impact,
            "energy": energies,
            "original_query": query
        }

