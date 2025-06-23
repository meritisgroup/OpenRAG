# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:37:47 2025

@author: chardy
"""
from ...utils.agent import get_Agent
from ...base_classes import RagAgent
from .prompts import prompts
import numpy as np
from ...database.database_class import get_database


class NaiveChatbot(RagAgent):

    def __init__(self, config_server: dict, *args) -> None:

        self.language = config_server["language"]
        self.agent = get_Agent(config_server)
        self.prompts = prompts[self.language]
        self.nb_chunks = 0
        self.db = get_database("chatbot_dummy_db", storage_path="./storage")

    def get_nb_token_embeddings(self):
        return 0

    def indexation_phase(
        self,
        path_input: str,
        reset_index: bool = False,
        chunk_size: int = 500,
        overlap: bool = True,
    ) -> None:
        return None

    def get_rag_context(
        self, query: str, nb_chunks: int = 0
    ) -> list[str]:
        return ""

    def generate_answer(
        self,
        query: str,
        nb_chunks: int = 0
    ) -> str:

        agent = self.agent
        nb_input_tokens = 0
        nb_output_tokens = 0

        prompt = self.prompts["smooth_generation"]["QUERY_TEMPLATE"].format(query=query)
        system_prompt = self.prompts["smooth_generation"]["SYSTEM_PROMPT"]
        answer = agent.predict(prompt=prompt, system_prompt=system_prompt)
        nb_input_tokens += np.sum(answer["nb_input_tokens"])
        nb_output_tokens += np.sum(answer["nb_output_tokens"])
        return {
            "answer": answer["texts"],
            "nb_input_tokens": nb_input_tokens,
            "nb_output_tokens": nb_output_tokens,
            "context": "",
            "impacts": answer["impacts"],
            "energy": answer["energy"],
        }

    def release_gpu_memory(self):
        self.agent.release_memory()

    def get_rag_contexts(
        self, queries: list[str], nb_chunks: int = 0
    ):
        contexts = []
        for query in queries:
            contexts.append("")
        return contexts

    def generate_answers(self, queries: list[str], nb_chunks=0):
        answers = []
        for query in queries:
            answer = self.generate_answer(query=query)
            answers.append(answer)
        return answers
