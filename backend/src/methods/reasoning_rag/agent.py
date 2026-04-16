from .indexation import ReasoningRagIndexation
from .search import ReasoningSearch
from application.agents.base_rag_agent import BaseRAGAgent
from utils.agent_functions import get_system_prompt
from .prompts import PROMPTS
from database.rag_classes import Section, Tokens
from core.error_handler import handle_errors, LLMError


class ReasoningRagAgent(BaseRAGAgent):

    def __init__(self, config_server: dict, models_infos: dict, dbs_name: list[str], data_folders_name: list[str]) -> None:
        super().__init__(config_server=config_server, models_infos=models_infos, dbs_name=dbs_name, data_folders_name=data_folders_name, rag_name='reasoning')
        self.data_manager.add_table(Section)
        self.data_manager.add_table(Tokens)
        self.prompts = PROMPTS[self.language]
        self.system_prompt = self._get_system_prompt(self.prompts)

    def indexation_phase(self, reset_index: bool = False, reset_preprocess: bool = False, overlap: bool = True, progress_callback=None, **kwargs) -> None:
        if reset_preprocess:
            reset_index = True
        if reset_index:
            for db_name in self.dbs_name:
                self.data_manager.delete_collection(vb_name=db_name, name='reasoning_rag')
            self.data_manager.clean_database()
        index = ReasoningRagIndexation(
            data_manager=self.data_manager,
            storage_path=self.storage_path,
            language=self.language,
            agent=self.agent,
            type_text_splitter=self.type_text_splitter,
            data_preprocessing=self.config_server['data_preprocessing'],
            embedding_model=self.embedding_model,
            llm_model=self.llm_model,
        )
        index.run_pipeline(
            chunk_size=self.chunk_size,
            chunk_overlap=overlap,
            config_server=self.config_server,
            reset_preprocess=reset_preprocess,
            progress_callback=progress_callback,
        )

    def get_rag_context(self, query: str, nb_chunks: int = 5):
        search = ReasoningSearch(
            agent=self.agent,
            model=self.llm_model,
            data_manager=self.data_manager,
            language=self.language,
            nb_chunks=nb_chunks,
        )
        return search.get_context(query=query)

    @handle_errors(reraise=True, exception_types=(LLMError,))
    def generate_answer(self, query: str, nb_chunks: int = 5, options_generation=None) -> dict:
        (impacts, energies) = ([0, 0, ''], [0, 0, ''])
        if self.reformulate_query:
            (query, input_t, output_t, impacts, energies) = self._reformulate_query_if_needed(query=query, nb_reformulation=1)
        (context, chunks, tokens_counter) = self.get_rag_context(query=query, nb_chunks=nb_chunks)
        self.add_tokens(tokens_counter['nb_input_tokens'], tokens_counter['nb_output_tokens'])
        prompt_template = self.prompts['smooth_generation']['QUERY_TEMPLATE']
        prompt = prompt_template.format(context=context, query=query)
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
