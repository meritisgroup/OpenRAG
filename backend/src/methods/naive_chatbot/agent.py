from methods.naive_rag.agent import NaiveRagAgent
from database.database_class import DataBase
from .prompts import prompts
import numpy as np
from database.rag_classes import Chunk

class NaiveChatbot(NaiveRagAgent):

    def __init__(self, config_server: dict, models_infos: dict, dbs_name=None, data_folders_name=None) -> None:
        if dbs_name is None:
            dbs_name = ['chatbot_dummy_db']
        if data_folders_name is None:
            data_folders_name = ['./storage']
        super().__init__(config_server=config_server, models_infos=models_infos, dbs_name=dbs_name, data_folders_name=data_folders_name)
        self.nb_chunks = 0
        self.db = DataBase(db_name='chatbot_dummy_db', path='./storage', path_data='.')
        self.prompts = prompts[self.language]
        self.system_prompt = self._get_system_prompt(self.prompts)

    def get_nb_token_embeddings(self):
        return 0

    def get_infos_embeddings(self):
        infos = {}
        infos['embedding_tokens'] = 0
        infos['input_tokens'] = 0
        infos['output_tokens'] = 0
        return infos

    def indexation_phase(self, reset_index: bool=False, overlap: bool=True, reset_preprocess=False) -> None:
        return None

    def get_rag_context(self, query: str, nb_chunks: int=0) -> list[str]:
        return [[]]

    def build_final_prompt(self, chunk_lists: list[list[Chunk]], query: str):
        prompt = self.prompts['smooth_generation']['QUERY_TEMPLATE'].format(query=query)
        return prompt

    def generate_answer(self, query: str, nb_chunks: int=0, options_generation=None) -> dict:
        chunks = self.get_rag_context(query=query, nb_chunks=nb_chunks)
        prompt = self.build_final_prompt(chunks, query)
        if options_generation is None:
            options_generation = self.config_server['options_generation']
        answer = self.agent.predict(prompt=prompt, system_prompt=self.system_prompt, options_generation=options_generation, model=self.llm_model)
        self.aggregate_response_tokens(answer)
        return self._build_response(answer_text=answer['texts'], context=[], query=query)

    def release_gpu_memory(self):
        self.agent.release_memory()

    def get_rag_contexts(self, queries: list[str], nb_chunks: int=0):
        contexts = []
        for query in queries:
            contexts.append('')
        return contexts