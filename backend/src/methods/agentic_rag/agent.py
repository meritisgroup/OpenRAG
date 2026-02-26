from base_classes import RagAgent
from methods.advanced_rag.agent import AdvancedRag
from methods.advanced_rag.agent import AdvancedRag
from database.database_class import get_management_data
from utils.agent import get_Agent
from .prompts import prompts
import numpy as np
from pydantic import BaseModel
from methods.query_reformulation.query_reformulation import query_reformulation

class CompareQueryAnswer(BaseModel):
    Decision: bool

class AgenticRagAgent(AdvancedRag):

    def __init__(self, config_server: dict, models_infos: dict, dbs_name: list[str], data_folders_name: list[str]) -> None:
        super().__init__(config_server=config_server, models_infos=models_infos, dbs_name=dbs_name, data_folders_name=data_folders_name)
        self.prompts = prompts[self.language]

    def evaluate(self, query: str, answer: str, agent) -> bool:
        system_prompt = self.prompts['evaluate']['SYSTEM_PROMPT']
        user_prompt = self.prompts['evaluate']['QUERY_TEMPLATE'].replace('{query}', str(query)).replace('{answer}', str(answer))
        result = agent.predict_json(system_prompt=system_prompt, model=self.llm_model, prompt=user_prompt, json_format=CompareQueryAnswer)
        return result.Decision

    def reformulate(self, query: str, answer: str, agent) -> str:
        system_prompt = self.prompts['reformulate']['SYSTEM_PROMPT']
        user_prompt = self.prompts['reformulate']['QUERY_TEMPLATE'].replace('{query}', str(query)).replace('{answer}', str(answer))
        new_query = agent.predict(system_prompt=system_prompt, prompt=user_prompt, model=self.llm_model)
        return new_query

    def concatene(self, answer_init: str, answer_add: str, query: str, agent, options_generation=None) -> str:
        system_prompt = self.prompts['concatenete']['SYSTEM_PROMPT']
        user_prompt = self.prompts['concatenete']['QUERY_TEMPLATE'].format(query=query, answer_init=answer_init, answer_add=answer_add)
        final_answer = agent.predict(system_prompt=system_prompt, prompt=user_prompt, model=self.llm_model, options_generation=options_generation)
        return final_answer['texts']

    def generate_answer(self, query: str, nb_chunks: int=5, max_iter=0, options_generation=None) -> str:
        agent = self.agent
        iter = 0
        info = super().generate_answer(query, nb_chunks=nb_chunks, options_generation=options_generation)
        answer = info['answer']
        nb_input_tokens = info['nb_input_tokens']
        nb_output_tokens = info['nb_output_tokens']
        context_tot = info['context']
        (impacts, energies) = (info['impacts'], info['energy'])
        while iter <= max_iter and (not self.evaluate(query, answer, agent)):
            iter += 1
            query_additional = self.reformulate(query, answer, agent)['texts']
            info = super().generate_answer(query_additional, nb_chunks=nb_chunks, options_generation=options_generation)
            answer_additional = info['answer']
            nb_input_tokens += np.sum(info['nb_input_tokens'])
            nb_output_tokens += np.sum(info['nb_output_tokens'])
            context_tot += info['context']
            impacts[0] += info['impacts'][0]
            impacts[1] += info['impacts'][1]
            impacts[2] = info['impacts'][2]
            energies[0] += info['energy'][0]
            energies[1] += info['energy'][1]
            energies[2] = info['energy'][2]
            answer = self.concatene(answer, answer_additional, query, agent, options_generation=options_generation)
        return {'answer': answer, 'nb_input_tokens': nb_input_tokens, 'nb_output_tokens': nb_output_tokens, 'context': context_tot, 'impacts': impacts, 'energy': energies, 'original_query': query}