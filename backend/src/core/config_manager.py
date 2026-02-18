from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

@dataclass
class LLMConfig:
    model: str
    provider: str
    api_key: Optional[str] = None
    url: Optional[str] = None
    temperature: float = 0.0
    max_attempts: int = 5
    type: str = 'llm'

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
        return cls(model=data.get('model', ''), provider=data.get('provider', 'openai'), api_key=data.get('api_key'), url=data.get('url'), temperature=data.get('temperature', 0.0), max_attempts=data.get('max_attempts', 5), type=data.get('type', 'llm'))

    def to_dict(self) -> Dict[str, Any]:
        return {'model': self.model, 'provider': self.provider, 'api_key': self.api_key, 'url': self.url, 'temperature': self.temperature, 'max_attempts': self.max_attempts, 'type': self.type}

@dataclass
class VectorStoreConfig:
    backend: str
    url: str
    embedding_model: str
    auth: Optional[List[str]] = None
    batch: bool = True
    type_retrieval: str = 'embeddings'

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorStoreConfig':
        params = data.get('params_vectorbase', {})
        return cls(backend=params.get('backend', 'elasticsearch'), url=params.get('url', ''), embedding_model=data.get('embedding_model', ''), auth=params.get('auth'), batch=params.get('batch', True), type_retrieval=data.get('type_retrieval', 'embeddings'))

    def to_dict(self) -> Dict[str, Any]:
        return {'backend': self.backend, 'url': self.url, 'embedding_model': self.embedding_model, 'auth': self.auth, 'batch': self.batch, 'type_retrieval': self.type_retrieval}

@dataclass
class RAGAgentConfig:
    llm: LLMConfig
    vector_store: VectorStoreConfig
    reranker_model: Optional[str] = None
    model_for_image: Optional[str] = None
    chunk_size: int = 1024
    chunk_overlap: bool = True
    nb_chunks: int = 5
    nb_chunks_reranker: int = 200
    language: str = 'EN'
    text_splitter: str = 'TextSplitter'
    reformulate_query: bool = False
    data_preprocessing: str = 'md_without_images'
    processor_chunks: List[str] = field(default_factory=list)

    @classmethod
    def from_legacy_dict(cls, config_server: Dict[str, Any]) -> 'RAGAgentConfig':
        llm_config = LLMConfig(model=config_server.get('model', ''), provider=config_server.get('default_mode_provider', 'openai'), api_key=config_server.get('params_host_llm', {}).get('api_key'), url=config_server.get('params_host_llm', {}).get('url'), temperature=0.0, max_attempts=config_server.get('max_attempts', 5), type='llm')
        vector_store_config = VectorStoreConfig.from_dict(config_server)
        return cls(llm=llm_config, vector_store=vector_store_config, reranker_model=config_server.get('reranker_model'), model_for_image=config_server.get('model_for_image'), chunk_size=config_server.get('chunk_length', 1024), chunk_overlap=config_server.get('chunk_overlap', True), nb_chunks=config_server.get('nb_chunks', 5), nb_chunks_reranker=config_server.get('nb_chunks_reranker', 200), language=config_server.get('language', 'EN'), text_splitter=config_server.get('TextSplitter', 'TextSplitter'), reformulate_query=config_server.get('reformulate_query', False), data_preprocessing=config_server.get('data_preprocessing', 'md_without_images'), processor_chunks=config_server.get('ProcessorChunks', []))

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {'model': self.llm.model, 'embedding_model': self.vector_store.embedding_model, 'reranker_model': self.reranker_model, 'model_for_image': self.model_for_image, 'default_mode_provider': self.llm.provider, 'chunk_length': self.chunk_size, 'chunk_overlap': self.chunk_overlap, 'nb_chunks': self.nb_chunks, 'nb_chunks_reranker': self.nb_chunks_reranker, 'language': self.language, 'TextSplitter': self.text_splitter, 'reformulate_query': self.reformulate_query, 'data_preprocessing': self.data_preprocessing, 'ProcessorChunks': self.processor_chunks, 'type_retrieval': self.vector_store.type_retrieval, 'params_vectorbase': {'backend': self.vector_store.backend, 'url': self.vector_store.url, 'batch': self.vector_store.batch, 'auth': self.vector_store.auth}, 'max_attempts': self.llm.max_attempts, 'params_host_llm': {'url': self.llm.url, 'api_key': self.llm.api_key, 'type': self.llm.provider}}

@dataclass
class AppConfig:
    rag: RAGAgentConfig
    storage_path: str
    data_path: str
    max_workers: int = 10
    device: str = 'cpu'
    mode: str = 'default'
    local_params: Dict[str, Any] = field(default_factory=dict)
    all_system_prompt: Dict[str, str] = field(default_factory=dict)
    options_generation: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, config_path: str='data/base_config_server.json') -> 'AppConfig':
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f'Configuration file not found: {config_path}')
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        required_fields = ['storage_path', 'storage_data_path', 'model', 'embedding_model']
        for field_name in required_fields:
            if field_name not in config_data:
                raise ValueError(f'Missing required configuration field: {field_name}')
        rag_config = RAGAgentConfig.from_legacy_dict(config_data)
        return cls(rag=rag_config, storage_path=config_data.get('storage_path', './storage'), data_path=config_data.get('storage_data_path', './data/databases'), max_workers=config_data.get('max_workers', 10), device=config_data.get('device', 'cpu'), mode=config_data.get('mode', 'default'), local_params=config_data.get('local_params', {}), all_system_prompt=config_data.get('all_system_prompt', {}), options_generation=config_data.get('options_generation', {}))

    def to_legacy_dict(self) -> Dict[str, Any]:
        config = self.rag.to_legacy_dict()
        config.update({'storage_path': self.storage_path, 'storage_data_path': self.data_path, 'max_workers': self.max_workers, 'device': self.device, 'mode': self.mode, 'local_params': self.local_params, 'all_system_prompt': self.all_system_prompt, 'options_generation': self.options_generation})
        return config

    def save(self, config_path: str='data/base_config_server.json') -> None:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_legacy_dict(), f, indent=4, ensure_ascii=False)

def load_config(config_path: str='data/base_config_server.json') -> AppConfig:
    return AppConfig.from_json(config_path)