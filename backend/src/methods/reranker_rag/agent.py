from methods.advanced_rag.agent import AdvancedRag
from methods.naive_rag.agent import NaiveRagAgent
from methods.naive_rag.query import NaiveSearch
from methods.naive_rag.indexation import contexts_to_prompts
import numpy as np
from itertools import chain
from database.rag_classes import Chunk
from utils.chunk_lists_merger import merge_chunk_lists

class RerankerRag(AdvancedRag):

    def __init__(self, config_server: dict, models_infos: dict, dbs_name: list[str], data_folders_name: list[str], type_processor_chunks: list[str]=[]) -> None:
        config_server['reformulate_query'] = False
        config_server['ProcessorChunks'] = []
        super().__init__(config_server=config_server, models_infos=models_infos, dbs_name=dbs_name, data_folders_name=data_folders_name)

    def get_rag_context(self, query: str, nb_chunks: int=5) -> list[list[Chunk]]:
        ns = NaiveSearch(data_manager=self.data_manager, nb_chunks=nb_chunks)
        chunk_lists = ns.get_context(query=query)
        return chunk_lists

    def release_gpu_memory(self):
        self.agent.release_memory()

    def generate_answer(self, query: str, nb_chunks: int=7, nb_reformulation=5, options_generation=None) -> str:
        agent = self.agent
        impacts = [0, 0, '']
        energy = [0, 0, '']
        if self.reformulate_query:
            (queries, input_t, output_t, impacts, energy) = self.reformulater.reformulate(query=query, nb_reformulation=nb_reformulation)
            self.nb_input_tokens += np.sum(input_t)
            self.nb_output_tokens += np.sum(output_t)
        else:
            queries = [query]
        chunk_lists = self.get_rag_context(query=queries[0], nb_chunks=nb_chunks)
        chunk_list = self.chunk_lists_merger(chunk_lists)
        docs_name = [chunk.document for chunk in chunk_list]
        if len(chunk_list) > 0 and self.rerank:
            (rerank_chunk_list, additional_data, nb_input_tokens) = self.reranker.rerank(query=query, chunk_list=chunk_list, max_contexts=len(chunk_list), additional_data={'docs_name': docs_name})
            self.nb_input_tokens += np.sum(nb_input_tokens)
        else:
            rerank_chunk_list = chunk_list
        prompt = self.build_final_prompt(rerank_chunk_list, query)
        if options_generation is None:
            options_generation = self.config_server['options_generation']
        answer = agent.predict(prompt=prompt, system_prompt=self.system_prompt, model=self.llm_model, options_generation=options_generation)
        self.nb_input_tokens += np.sum(answer['nb_input_tokens'])
        self.nb_output_tokens += np.sum(answer['nb_output_tokens'])
        impact = answer['impacts']
        impact[0] += impacts[0]
        impact[1] += impacts[1]
        energies = answer['energy']
        energies[0] += energy[0]
        energies[1] += energy[1]
        return {'answer': answer['texts'], 'nb_input_tokens': self.nb_input_tokens, 'nb_output_tokens': self.nb_output_tokens, 'context': rerank_chunk_list, 'impacts': impact, 'energy': energies, 'original_query': query}