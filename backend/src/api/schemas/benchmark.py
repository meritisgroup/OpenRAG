from pydantic import BaseModel
from typing import Dict, Any, Optional, List


class BenchmarkStartRequest(BaseModel):
    rag_names: List[str]
    databases: List[str]
    queries_doc_name: str
    config: Dict[str, Any]
    models_infos: Dict[str, Any]
    benchmark_type: str = 'full_bench'
    reset_index: bool = False
    reset_preprocess: bool = False


class BenchmarkStatus(BaseModel):
    benchmark_id: str
    status: str
    progress: float
    current_step: str
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class BenchmarkCompleteResult(BaseModel):
    benchmark_id: str
    status: str = "completed"
    scores: Dict[str, Any]
    files: Dict[str, str]
    plots: Dict[str, Any]
    databases: List[str] = []
    rag_names: List[str] = []
    error: Optional[str] = None


class GenerateQueriesRequest(BaseModel):
    databases: List[str]
    n_questions: int
    config: Dict[str, Any]
    models_infos: Dict[str, Any]


class GenerateQueriesResponse(BaseModel):
    queries: List[str]
    answers: List[str]
    file_path: str


class BenchmarkRequest(BaseModel):
    session_ids: List[str]
    rag_names: List[str]
    databases: List[str]
    queries_doc_name: str
    config: Dict[str, Any]
    models_infos: Dict[str, Any]
    benchmark_type: str = 'all'


class BenchmarkResult(BaseModel):
    benchmark_id: str
    status: str
    rag_names: List[str]
    databases: List[str]
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BenchmarkReport(BaseModel):
    report_id: str
    created_at: str
    rag_names: List[str]
    databases: List[str]
    type: str


class BenchmarkProgress(BaseModel):
    benchmark_id: str
    progress: float
    current_step: str
    status: str
