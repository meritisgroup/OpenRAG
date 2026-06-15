import time
from typing import Dict, Any, List, Optional
from streamlit_.api_client import APIClient
from streamlit_.api_client.exceptions import APIError
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
                        config_server: Dict[str, Any], models_infos: Dict[str, Any],
                        validate_models: bool = True, create_new_session: bool = False,
                        max_retries: int = 2) -> str:
        """
        Crée un agent RAG avec validation optionnelle des modèles et retry sur erreurs transitoires
        """
        client = cls.get_client()
        
        if create_new_session:
            session_id = client.create_session()
        elif client.session_id is None:
            session_id = client.create_session()
        else:
            session_id = client.session_id
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = client.create_agent(
                    rag_method=rag_method,
                    config=config_server,
                    models_infos=models_infos,
                    databases=databases_name,
                    validate_models=validate_models,
                    session_id=session_id
                )
                return session_id
            except APIError as e:
                last_error = e
                error_data = e.args[0] if e.args else {}
                if isinstance(error_data, dict):
                    detail = error_data.get('detail', error_data)
                    if isinstance(detail, dict) and 'validation' in detail:
                        raise
                    status_code = error_data.get('status_code', 0)
                    if status_code in (404, 400):
                        raise
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
        
        raise last_error
    
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
    def get_indexation_status(cls, session_id: Optional[str] = None) -> Dict[str, Any]:
        client = cls.get_client()
        return client.get_indexation_status(session_id=session_id)
    
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
    
    @classmethod
    def get_rags_availability(cls, config: Dict[str, Any], models_infos: Dict[str, Any]) -> Dict[str, Any]:
        """Retourne la disponibilité de tous les RAGs selon les modèles configurés"""
        client = cls.get_client()
        return client.get_rags_availability(config=config, models_infos=models_infos)
