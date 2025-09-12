import numpy as np
from backend.database.rag_classes import Chunk


class Reranker:
    def __init__(self, agent, reranking_model: str):
        """
        Args:
            agent (Agent_VLLM or Agent_Ollama class) : agent used to rerank chunks
            reranking_model (str) : name of model passed to agent
        """
        self.agent = agent
        self.reranking_model = reranking_model

    def rerank(
        self, query, chunk_list: list[Chunk], additional_data={}, max_contexts=5
    ) -> list[Chunk]:
        """
        Computes scores for all retrieved chunks and returns the max_contexts best

        Args:
            query (str) : query retrieved context will be compared to
            contexts (list[str]) : list of retrieved chunks
            additional_data (dict) : Metadata for retrieved chunks
            max_contexts (int) : top k chunks to keep

        Returns
            context (list[str]) : top max_contexts chunks, ordered by relevance to the query
            additional_data (np.array[dict]) : Metadata for top chunks

        """
        
        scores = self.agent.reranking(
            query=query, chunk_list=chunk_list, model=self.reranking_model
        )

        nb_input_tokens = np.sum(scores["nb_input_tokens"])
        scores = scores["scores"]
        for i, chunk in enumerate(chunk_list):
            chunk.rerank_score = scores[i]
        ranking_index = np.argsort(scores)[::-1][:max_contexts]
        rerank_chunk_list = np.array(chunk_list)[ranking_index]
        rerank_chunk_list = rerank_chunk_list.tolist()

        for key in additional_data.keys():
            additional_data[key] = np.array(additional_data[key])[ranking_index]

        return rerank_chunk_list, additional_data, nb_input_tokens
