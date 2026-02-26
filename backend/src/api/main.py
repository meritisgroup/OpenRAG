from typing import Any, Dict
from datetime import datetime
import uuid
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from factory import RAGFactory
from core.error_handler import configure_logging

log_file = os.getenv('LOG_FILE')
if log_file:
    configure_logging(log_file=log_file)

agent_cache: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    agent_cache.clear()


app = FastAPI(
    title="OpenRAG API",
    description="API REST pour OpenRAG - Benchmark et utilisation de méthodes RAG",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_session() -> str:
    session_id = str(uuid.uuid4())
    agent_cache[session_id] = {
        'agent': None,
        'created_at': datetime.now(),
        'rag_method': None,
        'databases': None
    }
    return session_id


def get_agent(session_id: str) -> Any:
    if session_id not in agent_cache:
        raise ValueError(f"Session {session_id} not found")
    return agent_cache[session_id].get('agent')


def set_agent(session_id: str, agent: Any, rag_method: str = None, databases: list = None) -> None:
    if session_id not in agent_cache:
        agent_cache[session_id] = {
            'agent': None,
            'created_at': datetime.now(),
            'rag_method': None,
            'databases': None
        }
    agent_cache[session_id]['agent'] = agent
    agent_cache[session_id]['rag_method'] = rag_method
    agent_cache[session_id]['databases'] = databases


def delete_session(session_id: str) -> bool:
    if session_id in agent_cache:
        del agent_cache[session_id]
        return True
    return False


def session_exists(session_id: str) -> bool:
    return session_id in agent_cache


def get_session_info(session_id: str) -> Dict[str, Any]:
    if session_id not in agent_cache:
        return None
    return {
        'session_id': session_id,
        'created_at': agent_cache[session_id]['created_at'],
        'has_agent': agent_cache[session_id]['agent'] is not None,
        'rag_method': agent_cache[session_id].get('rag_method'),
        'databases': agent_cache[session_id].get('databases')
    }


from .routers import (
    session_router, rag_router, config_router, databases_router, 
    documents_router, benchmark_router, queries_router, storage_router
)

app.include_router(session_router, prefix="/api/session", tags=["Session"])
app.include_router(rag_router, prefix="/api/rag", tags=["RAG"])
app.include_router(config_router, prefix="/api/config", tags=["Configuration"])
app.include_router(databases_router, prefix="/api/databases", tags=["Databases"])
app.include_router(documents_router, prefix="/api/documents", tags=["Documents"])
app.include_router(benchmark_router, prefix="/api/benchmark", tags=["Benchmark"])
app.include_router(queries_router, prefix="/api/queries", tags=["Queries"])
app.include_router(storage_router, prefix="/api/storage", tags=["Storage"])


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/info", tags=["Info"])
def get_system_info():
    return {
        "rag_methods": RAGFactory.list_available_rags(),
        "active_sessions": len(agent_cache)
    }
