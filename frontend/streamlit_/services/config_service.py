from typing import Dict, Any, Optional
from streamlit_.api_client import APIClient
from streamlit_.core.config import API_BASE_URL


class ConfigService:
    _client: Optional[APIClient] = None
    
    @classmethod
    def get_client(cls) -> APIClient:
        if cls._client is None:
            cls._client = APIClient(API_BASE_URL)
        return cls._client
    
    @classmethod
    def set_client(cls, client: APIClient) -> None:
        cls._client = client
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        client = cls.get_client()
        return client.get_config()
    
    @classmethod
    def update_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        client = cls.get_client()
        return client.update_config(config)
    
    @classmethod
    def update_local_params(cls, forced_system_prompt: bool = False,
                            generation_system_prompt_name: str = 'default') -> Dict[str, Any]:
        client = cls.get_client()
        return client.update_local_params(
            forced_system_prompt=forced_system_prompt,
            generation_system_prompt_name=generation_system_prompt_name
        )
    
    @classmethod
    def reset_local_params(cls) -> Dict[str, Any]:
        client = cls.get_client()
        return client.reset_local_params()
    
    @classmethod
    def get_system_info(cls) -> Dict[str, Any]:
        client = cls.get_client()
        return client.get_system_info()
    
    @classmethod
    def change_server_config(cls, rag_name: Optional[str] = None, 
                             mode: str = 'Simple') -> Dict[str, Any]:
        client = cls.get_client()
        return client.change_server_config(rag_name=rag_name, mode=mode)
    
    @classmethod
    def get_models(cls) -> Dict[str, Any]:
        client = cls.get_client()
        return client.get_models()
    
    @classmethod
    def update_models(cls, models: Dict[str, Any]) -> Dict[str, Any]:
        client = cls.get_client()
        return client.update_models(models)
    
    @classmethod
    def get_providers(cls) -> Dict[str, Any]:
        client = cls.get_client()
        return client.get_providers()
    
    @classmethod
    def update_providers(cls, providers: Dict[str, Any]) -> Dict[str, Any]:
        client = cls.get_client()
        return client.update_providers(providers)
    
    @classmethod
    def get_all_rags(cls) -> Dict[str, Any]:
        client = cls.get_client()
        return client.get_all_rags()
    
    @classmethod
    def update_all_rags(cls, all_rags: Dict[str, Any]) -> Dict[str, Any]:
        client = cls.get_client()
        return client.update_all_rags(all_rags)
