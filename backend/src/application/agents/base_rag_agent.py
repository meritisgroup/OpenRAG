from typing import List, Dict, Any, Optional
import numpy as np
from base_classes import RagAgent
from core.mixins.token_counter_mixin import TokenCounterMixin
from core.response_builder import RAGResponseBuilder, create_rag_response
from core.config_manager import RAGAgentConfig
from core.error_handler import RAGError, handle_errors, LLMError, DatabaseError, VectorStoreError
from database.rag_classes import Chunk, Document
from infrastructure.llm.llm_provider_factory import LLMProviderFactory
from infrastructure.llm.base_provider import BaseLLMProvider
from utils.agent_functions import get_system_prompt
from infrastructure.database.coordinated_data_manager import get_management_data
from methods.query_reformulation.query_reformulation import query_reformulation

class BaseRAGAgent(RagAgent, TokenCounterMixin):

    def __init__(self, config_server: Dict[str, Any], models_infos: Dict[str, Any], dbs_name: List[str], data_folders_name: List[str], rag_name: str='base'):
        TokenCounterMixin.__init__(self)
        self.config_server = config_server
        self.models_infos = models_infos
        self.dbs_name = dbs_name
        self.data_folders_name = data_folders_name
        self.rag_name = rag_name
        self.llm_model = config_server.get('model', '')
        self.embedding_model = config_server.get('embedding_model', '')
        self.storage_path = config_server.get('storage_path', './storage')
        self.nb_chunks = config_server.get('nb_chunks', 5)
        self.language = config_server.get('language', 'EN')
        self.type_text_splitter = config_server.get('TextSplitter', 'TextSplitter')
        self.type_retrieval = config_server.get('type_retrieval', 'embeddings')
        self.chunk_size = config_server.get('chunk_length', 1024)
        self.reformulate_query = config_server.get('reformulate_query', False)
        self.providers = LLMProviderFactory.create_all_providers(models_infos=models_infos, language=self.language, max_attempts=config_server.get('max_attempts', 5), max_workers=config_server.get('max_workers', 10))
        if self.llm_model in self.providers:
            self.agent = self.providers[self.llm_model]
        elif len(self.providers) > 0:
            first_model = next(iter(self.providers))
            self.agent = self.providers[first_model]
            print(f"Warning: Model '{self.llm_model}' not found in providers, using '{first_model}'")
        else:
            self.agent = BaseLLMProvider(models_infos=models_infos, language=self.language, max_attempts=config_server.get('max_attempts', 5), max_workers=config_server.get('max_workers', 10))
            self.providers = {self.llm_model: self.agent}
        self.data_manager = get_management_data(dbs_name=self.dbs_name, data_folders_name=self.data_folders_name, storage_path=self.storage_path, config_server=config_server, agent=self.agent)
        self.response_builder = RAGResponseBuilder()
        self.reformulater = None
        if self.reformulate_query:
            self._init_query_reformulation()
        self.prompts = {}

    def _init_query_reformulation(self) -> None:
        try:
            self.reformulater = query_reformulation(agent=self.agent, language=self.language, model=self.llm_model)
        except ImportError:
            self.reformulate_query = False
            print('Warning: Query reformulation requested but module not available')

    @property
    def nb_input_tokens(self) -> int:
        return self._nb_input_tokens

    @nb_input_tokens.setter
    def nb_input_tokens(self, value: int) -> None:
        self._nb_input_tokens = value

    @property
    def nb_output_tokens(self) -> int:
        return self._nb_output_tokens

    @nb_output_tokens.setter
    def nb_output_tokens(self, value: int) -> None:
        self._nb_output_tokens = value

    @handle_errors(reraise=True, exception_types=(DatabaseError, VectorStoreError))
    def get_nb_token_embeddings(self) -> int:
        return self.data_manager.get_nb_token_embeddings()

    @handle_errors(reraise=True, exception_types=(DatabaseError,))
    def get_infos_embeddings(self) -> Dict[str, int]:
        infos = {}
        documents = self.data_manager.query(Document)
        infos['embedding_tokens'] = sum(doc.embedding_tokens or 0 for doc in documents)
        infos['input_tokens'] = sum(doc.input_tokens or 0 for doc in documents)
        infos['output_tokens'] = sum(doc.output_tokens or 0 for doc in documents)
        return infos

    def _build_response(self, answer_text: str, context: List[Chunk], query: str, additional_llm_calls: Optional[List[Dict[str, Any]]]=None, impacts: Optional[List]=None, energy: Optional[List]=None) -> Dict[str, Any]:
        builder = RAGResponseBuilder()
        builder.set_answer(answer_text)
        builder.set_tokens(self._nb_input_tokens, self._nb_output_tokens)
        builder.set_context(context)
        builder.set_query(query)
        if impacts:
            builder.set_impacts(impacts)
        if energy:
            builder.set_energy(energy)
        if additional_llm_calls:
            for llm_response in additional_llm_calls:
                builder.aggregate_llm_response(llm_response)
        return builder.build_dict()

    def _get_system_prompt(self, prompts: Dict[str, Any]) -> str:
        return get_system_prompt(self.config_server, prompts)

    @handle_errors(reraise=False, default_return=(None, 0, 0, [0, 0, ''], [0, 0, '']), exception_types=(LLMError,))
    def _reformulate_query_if_needed(self, query: str, nb_reformulation: int=1) -> tuple[str, int, int, List, List]:
        if not self.reformulate_query or self.reformulater is None:
            return (query, 0, 0, [0, 0, ''], [0, 0, ''])
        result = self.reformulater.reformulate(query=query, nb_reformulation=nb_reformulation)
        if len(result) >= 5:
            (queries, input_t, output_t, impacts, energies) = result[:5]
            reformulated_query = queries[0] if queries else query
            self.add_tokens(np.sum(input_t), np.sum(output_t))
            return (reformulated_query, np.sum(input_t), np.sum(output_t), impacts, energies)
        return (query, 0, 0, [0, 0, ''], [0, 0, ''])

    def release_gpu_memory(self) -> None:
        if hasattr(self.agent, 'release_memory'):
            self.agent.release_memory()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(model={self.llm_model}, dbs={len(self.dbs_name)}, tokens={self.total_tokens})'