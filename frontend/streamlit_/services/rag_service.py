from typing import Dict, Any, List, Optional
from streamlit_.api_client import APIClient
from streamlit_.core.config import API_BASE_URL


class RAGService:
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
    def create_session(cls) -> str:
        client = cls.get_client()
        return client.create_session()
    
    @classmethod
    def get_chat_agent(cls, rag_method: str, databases_name: List[str],
                        config_server: Dict[str, Any], models_infos: Dict[str, Any]) -> str:
        client = cls.get_client()
        if client.session_id is None:
            client.create_session()
        
        response = client.create_agent(
            rag_method=rag_method,
            config=config_server,
            models_infos=models_infos,
            databases=databases_name
        )
        return client.session_id
    
    @classmethod
    def run_indexation(cls, session_id: Optional[str] = None, reset_index: bool = False,
                       reset_preprocess: bool = False) -> Dict[str, Any]:
        client = cls.get_client()
        return client.run_indexation(
            reset_index=reset_index,
            reset_preprocess=reset_preprocess,
            session_id=session_id
        )
    
    @classmethod
    def generate_answer(cls, query: str, nb_chunks: int = 5,
                        options_generation: Optional[Dict[str, Any]] = None,
                        session_id: Optional[str] = None) -> Dict[str, Any]:
        client = cls.get_client()
        return client.generate_answer(
            query=query,
            nb_chunks=nb_chunks,
            options_generation=options_generation,
            session_id=session_id
        )
    
    @classmethod
    def list_methods(cls) -> List[Dict[str, str]]:
        client = cls.get_client()
        return client.list_rag_methods()
    
    @classmethod
    def get_agent_status(cls, session_id: Optional[str] = None) -> Dict[str, Any]:
        client = cls.get_client()
        return client.get_agent_status(session_id=session_id)
    
    @classmethod
    def delete_session(cls, session_id: Optional[str] = None) -> bool:
        client = cls.get_client()
        return client.delete_session(session_id=session_id)
