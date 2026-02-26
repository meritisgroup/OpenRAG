from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Union
from datetime import datetime


class SessionCreate(BaseModel):
    session_id: Optional[str] = None


class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    has_agent: bool
    rag_method: Optional[str] = None
    databases: Optional[List[str]] = None


class RAGMethod(BaseModel):
    id: str
    name: str
    description: Optional[str] = None


class CreateAgentRequest(BaseModel):
    session_id: str
    rag_method: str
    config: Dict[str, Any]
    models_infos: Dict[str, Any]
    databases: List[str] = ['']


class IndexRequest(BaseModel):
    session_id: str
    reset_index: bool = False
    reset_preprocess: bool = False


class GenerateRequest(BaseModel):
    session_id: str
    query: str
    nb_chunks: int = 5
    options_generation: Optional[Dict[str, Any]] = None


class ChunkInfo(BaseModel):
    text: str
    document: Optional[str] = None
    rerank_score: Optional[float] = None
    chunk_id: Optional[int] = None


class GenerateResponse(BaseModel):
    answer: str
    nb_input_tokens: int
    nb_output_tokens: int
    context: Union[List[ChunkInfo], str]
    impacts: List[Any]
    energy: List[Any]
    original_query: Optional[str] = None
    time: Optional[float] = None


class AgentStatus(BaseModel):
    session_id: str
    rag_method: str
    databases: List[str]
    total_tokens: int
    nb_input_tokens: int
    nb_output_tokens: int
