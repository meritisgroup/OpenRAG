from .indexation import NaiveRagIndexation
from .query import NaiveSearch
from application.agents.base_rag_agent import BaseRAGAgent
from .prompts import prompts
from utils.chunk_lists_merger import merge_chunk_lists
from database.rag_classes import Chunk
from .indexation import concat_chunks
from core.error_handler import handle_errors, LLMError, RetrievalError, VectorStoreError

class NaiveRagAgent(BaseRAGAgent):

    def __init__(self, config_server: dict, models_infos: dict, dbs_name: list[str], data_folders_name: list[str]) -> None:
        super().__init__(config_server=config_server, models_infos=models_infos, dbs_name=dbs_name, data_folders_name=data_folders_name, rag_name='naive')
        self.prompts = prompts[self.language]
        self.system_prompt = self._get_system_prompt(self.prompts)
        self.chunk_lists_merger = merge_chunk_lists

    def indexation_phase(self, reset_index: bool=False, reset_preprocess: bool=False, overlap: bool=True) -> None:
        if reset_preprocess:
            reset_index = True
        if reset_index:
            self.data_manager.delete_collection()
            self.data_manager.clean_database()
        index = NaiveRagIndexation(data_manager=self.data_manager, type_text_splitter=self.type_text_splitter, data_preprocessing=self.config_server['data_preprocessing'], agent=self.agent, embedding_model=self.embedding_model)
        index.run_pipeline(chunk_size=self.chunk_size, chunk_overlap=overlap, config_server=self.config_server, reset_preprocess=reset_preprocess, max_workers=self.config_server['max_workers'])
        return None

    def get_rag_context(self, query: str, nb_chunks: int=5) -> list[list[Chunk]]:
        ns = NaiveSearch(data_manager=self.data_manager, nb_chunks=nb_chunks)
        return ns.get_context(query=query)

    def build_final_prompt(self, chunk_list: list[Chunk], query: str) -> str:
        context = concat_chunks(chunk_list)
        prompt = self.prompts['smooth_generation']['QUERY_TEMPLATE'].format(context=context, query=query)
        return prompt

    @handle_errors(reraise=True, exception_types=(LLMError, RetrievalError, VectorStoreError))
    def generate_answer(self, query: str, nb_chunks: int=5, options_generation=None) -> dict:
        (impacts, energies) = ([0, 0, ''], [0, 0, ''])
        if self.reformulate_query:
            (query, input_t, output_t, impacts, energies) = self._reformulate_query_if_needed(query=query, nb_reformulation=1)
        chunk_lists = self.get_rag_context(query=query, nb_chunks=nb_chunks)
        merged_chunk_list = self.chunk_lists_merger(chunk_lists)
        prompt = self.build_final_prompt(merged_chunk_list, query)
        chunks = [chunk for chunk_list in chunk_lists for chunk in chunk_list]
        if options_generation is None:
            options_generation = self.config_server['options_generation']
        answer = self.agent.predict(prompt=prompt, system_prompt=self.system_prompt, options_generation=options_generation, model=self.llm_model)
        self.aggregate_response_tokens(answer)
        impacts[2] = answer['impacts'][2]
        impacts[0] += answer['impacts'][0]
        impacts[1] += answer['impacts'][1]
        energies[2] = answer['energy'][2]
        energies[0] += answer['energy'][0]
        energies[1] += answer['energy'][1]
        return self._build_response(answer_text=answer['texts'], context=chunks, query=query, impacts=impacts, energy=energies)

    def release_gpu_memory(self):
        if hasattr(self.agent, 'release_memory'):
            self.agent.release_memory()