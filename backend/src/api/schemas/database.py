from pydantic import BaseModel
from typing import Dict, Any, Optional, List


class DatabaseInfo(BaseModel):
    name: str
    path: str
    document_count: int
    created_at: Optional[str] = None


class DatabaseCreateRequest(BaseModel):
    name: str
    embedding_model: str
    documents: Optional[List[str]] = None


class DatabaseListResponse(BaseModel):
    databases: List[DatabaseInfo]


class DocumentInfo(BaseModel):
    id: str
    name: str
    path: str
    chunk_count: int
    embedding_tokens: int


class DocumentUploadResponse(BaseModel):
    document_id: str
    name: str
    status: str
    chunk_count: int


class MetadatasResponse(BaseModel):
    database_name: str
    metadatas: Dict[str, Any]


class MetadatasUpdateRequest(BaseModel):
    metadatas: Dict[str, Any]


class DatabaseDocumentUploadResponse(BaseModel):
    database_name: str
    document_name: str
    status: str
    path: str
