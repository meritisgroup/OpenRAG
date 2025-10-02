# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:37:47 2025

@author: chardy
"""
from ...utils.agent import get_Agent
from ...utils.agent_functions import get_system_prompt
from ..naive_rag.agent import NaiveRagAgent
from ...database.database_class import get_management_data
from .prompts import prompts
import numpy as np
from ...database.database_class import DataBase
from backend.database.rag_classes import Chunk


class NaiveChatbot(NaiveRagAgent):

    def __init__(self, config_server: dict, *args) -> None:

        self.language = config_server["language"]
        self.agent = get_Agent(config_server)
        self.config_server = config_server

        self.prompts = prompts[self.language]
        self.system_prompt = get_system_prompt(config_server, self.prompts)

        self.nb_chunks = 0
        self.db = DataBase(db_name="chatbot_dummy_db", path="./storage", path_data=".")

        self.data_manager = get_management_data(
            dbs_name=["chatbot_dummy_db"],
            data_folders_name=["./storage"],
            storage_path=config_server["storage_path"],
            config_server=config_server,
            agent=self.agent,
        )

    def get_nb_token_embeddings(self):
        return 0

    def get_infos_embeddings(self):
        infos = {}
        infos["embedding_tokens"] = 0
        infos["input_tokens"] = 0
        infos["output_tokens"] = 0
        return infos

    def indexation_phase(
        self,
        reset_index: bool = False,
        overlap: bool = True,
        reset_preprocess = False
    ) -> None:
        return None

    def get_rag_context(self, query: str, nb_chunks: int = 0) -> list[str]:
        return [[]]

    def build_final_prompt(self, chunk_lists: list[list[Chunk]], query: str):
        prompt = self.prompts["smooth_generation"]["QUERY_TEMPLATE"].format(query=query)
        return prompt

    def generate_answer(
        self, query: str, nb_chunks: int = 0, options_generation=None
    ) -> str:

        agent = self.agent
        nb_input_tokens = 0
        nb_output_tokens = 0

        chunks = self.get_rag_context(query=query, nb_chunks=nb_chunks)
        prompt = self.build_final_prompt(chunks, query)

        if options_generation is None:
            options_generation = self.config_server["options_generation"]

        answer = agent.predict(
            prompt=prompt,
            system_prompt=self.system_prompt,
            options_generation=options_generation,
        )
        nb_input_tokens += np.sum(answer["nb_input_tokens"])
        nb_output_tokens += np.sum(answer["nb_output_tokens"])
        return {
            "answer": answer["texts"],
            "nb_input_tokens": nb_input_tokens,
            "nb_output_tokens": nb_output_tokens,
            "context": [],
            "impacts": answer["impacts"],
            "energy": answer["energy"],
        }

    def release_gpu_memory(self):
        self.agent.release_memory()

    def get_rag_contexts(self, queries: list[str], nb_chunks: int = 0):
        contexts = []
        for query in queries:
            contexts.append("")
        return contexts

