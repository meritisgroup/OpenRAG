import numpy as np


class Reranker:
    def __init__(self, agent, reranking_model: str):
        """
        Args:
            agent (Agent_VLLM or Agent_Ollama class) : agent used to rerank chunks
            reranking_model (str) : name of model passed to agent
        """
        self.agent = agent
        self.reranking_model = reranking_model

    def rerank(self, query, contexts, additional_data={}, max_contexts=5):
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
            query=query, contexts=contexts, model=self.reranking_model
        )

        nb_input_tokens = np.sum(scores["nb_input_tokens"])
        scores = scores["scores"]
        ranking_index = np.argsort(scores)[::-1][:max_contexts]
        contexts = np.array(contexts)[ranking_index]

        for key in additional_data.keys():
            additional_data[key] = np.array(additional_data[key])[ranking_index]
        return contexts, additional_data, nb_input_tokens
