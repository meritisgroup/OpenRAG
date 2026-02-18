import os
import re
from typing import Dict, List, Any, Optional, Type, Union
from .rag_registry import RAGConfig, RAG_REGISTRY
from utils.factory_name_dataset_vectorbase import get_name


class RAGFactory:
    _instance = None
    _registry: Dict[str, RAGConfig]

    def __new__(cls, registry: Optional[Dict[str, RAGConfig]] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = registry or RAG_REGISTRY.copy()
        return cls._instance

    def get_agent(
        self,
        rag_name: str,
        config_server: Dict[str, Any],
        models_infos: Dict[str, Any],
        databases_name: Optional[List[str]] = None,
        custom: bool = False
    ):
        rag_name = self._normalize_name(rag_name)
        config = self._get_rag_config(rag_name)
        
        if custom and not config.supported_in_custom:
            raise ValueError(f"RAG '{rag_name}' does not support custom mode")
        
        dbs_name, data_folders = self._prepare_database_names(
            rag_name, config_server, databases_name or [''], custom
        )
        
        kwargs = {
            'config_server': config_server,
            'models_infos': models_infos,
            'dbs_name': dbs_name,
            'data_folders_name': data_folders
        }
        
        return config.agent_class(**kwargs)

    def _get_rag_config(self, rag_name: str) -> RAGConfig:
        if rag_name in self._registry:
            return self._registry[rag_name]
        
        for name, config in self._registry.items():
            if rag_name in config.aliases:
                return config
        
        custom_rags_path = f'data/custom_rags/{rag_name}.json'
        if os.path.exists(custom_rags_path):
            import json
            with open(custom_rags_path, 'r') as f:
                custom_config = json.load(f)
            base_rag = custom_config.get('base', 'naive')
            if base_rag in self._registry:
                return self._registry[base_rag]
        
        raise ValueError(f"Unknown RAG method: '{rag_name}'. Available: {self.list_available_rags()}")

    def _normalize_name(self, name: str) -> str:
        return name.lower().strip()

    def _prepare_database_names(
        self,
        rag_name: str,
        config_server: Dict[str, Any],
        databases_name: List[str],
        custom: bool
    ):
        if custom:
            return self._prepare_custom_names(config_server, databases_name)
        return self._prepare_standard_names(rag_name, config_server, databases_name)

    def _prepare_standard_names(
        self,
        rag_name: str,
        config_server: Dict[str, Any],
        databases_name: List[str]
    ):
        names = []
        extended_databases = []
        
        for db_name in databases_name:
            generated_names = get_name(
                rag_name=rag_name,
                config_server=config_server,
                additionnal_name=db_name
            )
            extended_databases.extend([db_name] * len(generated_names))
            names.extend(generated_names)
        
        embedding_model = config_server.get('embedding_model', '')
        if isinstance(embedding_model, list):
            embedding_model = self._flatten_embedding_models(embedding_model, len(databases_name))
            config_server['embedding_model'] = embedding_model
        
        return names, extended_databases

    def _prepare_custom_names(
        self,
        config_server: Dict[str, Any],
        databases_name: List[str]
    ):
        base_name = config_server.get('name', '')
        embedding_model = config_server.get('embedding_model', '')
        safe_embedding = re.sub('[^a-zA-Z0-9]', '_', str(embedding_model))
        
        names = [
            f"{base_name}___{safe_embedding}___{db_name}".lower()
            for db_name in databases_name
        ]
        
        return names, databases_name

    def _flatten_embedding_models(self, embedding_models: List, db_count: int) -> List:
        flattened = [embedding_models.copy()] * db_count
        return [item for sublist in flattened for item in sublist]

    @classmethod
    def list_available_rags(cls) -> List[str]:
        rags = list(RAG_REGISTRY.keys())
        custom_rags_path = 'data/custom_rags'
        if os.path.exists(custom_rags_path):
            for f in os.listdir(custom_rags_path):
                if f.endswith('.json'):
                    rags.append(f.replace('.json', ''))
        return rags

    @classmethod
    def register(cls, name: str, config: RAGConfig) -> None:
        instance = cls()
        instance._registry[name] = config

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
