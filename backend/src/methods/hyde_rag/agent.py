from .indexation import NaiveRagIndexation, concat_chunks
from .query import HydeSearch
from application.agents.base_rag_agent import BaseRAGAgent
from .prompts import prompts
from .hypothetical_document import HypotheticalDocumentGenerator
from utils.chunk_lists_merger import merge_chunk_lists
from database.rag_classes import Chunk
from core.error_handler import handle_errors, LLMError, RetrievalError, VectorStoreError

class HydeRagAgent(BaseRAGAgent):

    def __init__(self, config_server: dict, models_infos: dict, dbs_name: list[str], data_folders_name: list[str]) -> None:
        super().__init__(config_server=config_server, models_infos=models_infos, dbs_name=dbs_name, data_folders_name=data_folders_name, rag_name='hyde')
        self.prompts = prompts[self.language]
        self.system_prompt = self._get_system_prompt(self.prompts)
        self.chunk_lists_merger = merge_chunk_lists

        # Initialize hypothetical document generator
        self.doc_generator = HypotheticalDocumentGenerator(
            agent=self.agent,
            model=self.llm_model,
            language=self.language
        )

    def indexation_phase(self, reset_index: bool=False, reset_preprocess: bool=False, overlap: bool=True, progress_callback=None, **kwargs) -> None:
        if reset_preprocess:
            reset_index = True
        if reset_index:
            self.data_manager.delete_collection()
            self.data_manager.clean_database()
        index = NaiveRagIndexation(
            data_manager=self.data_manager,
            type_text_splitter=self.type_text_splitter,
            data_preprocessing=self.config_server['data_preprocessing'],
            agent=self.agent,
            embedding_model=self.embedding_model
        )
        index.run_pipeline(
            chunk_size=self.chunk_size,
            chunk_overlap=overlap,
            config_server=self.config_server,
            reset_preprocess=reset_preprocess,
            max_workers=self.config_server['max_workers'],
            progress_callback=progress_callback
        )
        return None

    def get_rag_context(self, query: str, nb_chunks: int=5) -> tuple[list[list[Chunk]], dict]:
        """
        Get RAG context using HyDE approach.
        Returns (chunk_lists, metadata) where metadata contains info about hypothetical documents.
        """
        hs = HydeSearch(
            data_manager=self.data_manager,
            nb_chunks=nb_chunks,
            doc_generator=self.doc_generator
        )
        return hs.get_context(query=query)

    def build_final_prompt(self, chunk_list: list[Chunk], query: str) -> str:
        context = concat_chunks(chunk_list)
        prompt = self.prompts['smooth_generation']['QUERY_TEMPLATE'].format(context=context, query=query)
        return prompt

    @handle_errors(reraise=True, exception_types=(LLMError, RetrievalError, VectorStoreError))
    def generate_answer(self, query: str, nb_chunks: int=5, options_generation=None) -> dict:
        # Initialize impacts and energy tracking
        (impacts, energies) = ([0, 0, ''], [0, 0, ''])

        # Handle query reformulation if enabled
        if self.reformulate_query:
            (query, input_t, output_t, impacts, energies) = self._reformulate_query_if_needed(query=query, nb_reformulation=1)

        # Get context using HyDE approach
        (chunk_lists, hyde_metadata) = self.get_rag_context(query=query, nb_chunks=nb_chunks)

        # Aggregate HyDE metadata
        impacts[0] += hyde_metadata['impacts'][0]
        impacts[1] += hyde_metadata['impacts'][1]
        impacts[2] = hyde_metadata['impacts'][2]
        energies[0] += hyde_metadata['energy'][0]
        energies[1] += hyde_metadata['energy'][1]
        energies[2] = hyde_metadata['energy'][2]

        # Merge chunks from different sources
        merged_chunk_list = self.chunk_lists_merger(chunk_lists)

        # Build final prompt with retrieved context
        prompt = self.build_final_prompt(merged_chunk_list, query)

        # Flatten chunks for response
        chunks = [chunk for chunk_list in chunk_lists for chunk in chunk_list]

        # Set options for generation
        if options_generation is None:
            options_generation = self.config_server['options_generation']

        # Generate final answer
        answer = self.agent.predict(
            prompt=prompt,
            system_prompt=self.system_prompt,
            options_generation=options_generation,
            model=self.llm_model
        )

        # Aggregate response tokens
        self.aggregate_response_tokens(answer)

        # Add final answer impacts and energy
        impacts[2] = answer['impacts'][2]
        impacts[0] += answer['impacts'][0]
        impacts[1] += answer['impacts'][1]
        energies[2] = answer['energy'][2]
        energies[0] += answer['energy'][0]
        energies[1] += answer['energy'][1]

        # Build response with hypothetical documents included
        response = self._build_response(
            answer_text=answer['texts'],
            context=chunks,
            query=query,
            impacts=impacts,
            energy=energies
        )

        # Add HyDE metadata to response for transparency
        response['hyde_metadata'] = {
            'hypothetical_documents': hyde_metadata['hypothetical_documents'],
            'doc_generation_tokens': {
                'input': hyde_metadata['nb_input_tokens'],
                'output': hyde_metadata['nb_output_tokens']
            }
        }

        return response

    def release_gpu_memory(self):
        if hasattr(self.agent, 'release_memory'):
            self.agent.release_memory()
