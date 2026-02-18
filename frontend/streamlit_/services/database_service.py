from typing import Dict, Any, List, Optional
from streamlit_.api_client import APIClient
from streamlit_.core.config import API_BASE_URL


class DatabaseService:
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
    def list_databases(cls) -> List[Dict[str, Any]]:
        client = cls.get_client()
        return client.list_databases()
    
    @classmethod
    def create_database(cls, name: str, embedding_model: str = '') -> Dict[str, Any]:
        client = cls.get_client()
        return client.create_database(name=name, embedding_model=embedding_model)
    
    @classmethod
    def delete_database(cls, name: str) -> Dict[str, Any]:
        client = cls.get_client()
        return client.delete_database(name=name)
    
    @classmethod
    def get_database_documents(cls, name: str) -> List[str]:
        client = cls.get_client()
        return client.get_database_documents(name=name)
    
    @classmethod
    def upload_document(cls, file_path: str) -> Dict[str, Any]:
        client = cls.get_client()
        return client.upload_document(file_path=file_path)
    
    @classmethod
    def get_document_content(cls, document_id: str) -> str:
        client = cls.get_client()
        return client.get_document_content(document_id=document_id)
