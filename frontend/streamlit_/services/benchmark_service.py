import time
from typing import Dict, Any, List, Optional
from streamlit_.api_client import APIClient
from streamlit_.core.config import API_BASE_URL


class BenchmarkService:
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
    def list_reports(cls) -> List[Dict[str, Any]]:
        client = cls.get_client()
        return client.list_benchmark_reports()
    
    @classmethod
    def get_report(cls, report_id: str) -> Dict[str, Any]:
        client = cls.get_client()
        return client.get_benchmark_report(report_id=report_id)
    
    @classmethod
    def delete_report(cls, report_id: str) -> Dict[str, Any]:
        client = cls.get_client()
        return client.delete_benchmark_report(report_id=report_id)
    
    @classmethod
    def run_benchmark(cls, session_ids: List[str], rag_names: List[str],
                      databases: List[str], queries_doc_name: str,
                      config: Dict[str, Any], models_infos: Dict[str, Any],
                      benchmark_type: str = 'all') -> Dict[str, Any]:
        client = cls.get_client()
        return client.run_benchmark(
            session_ids=session_ids,
            rag_names=rag_names,
            databases=databases,
            queries_doc_name=queries_doc_name,
            config=config,
            models_infos=models_infos,
            benchmark_type=benchmark_type
        )
    
    @classmethod
    def start_benchmark(cls, rag_names: List[str], databases: List[str],
                        queries_doc_name: str, config: Dict[str, Any],
                        models_infos: Dict[str, Any], benchmark_type: str = 'full_bench',
                        reset_index: bool = False, reset_preprocess: bool = False) -> str:
        client = cls.get_client()
        result = client.start_benchmark(
            rag_names=rag_names,
            databases=databases,
            queries_doc_name=queries_doc_name,
            config=config,
            models_infos=models_infos,
            benchmark_type=benchmark_type,
            reset_index=reset_index,
            reset_preprocess=reset_preprocess
        )
        return result.get('benchmark_id', '')
    
    @classmethod
    def get_status(cls, benchmark_id: str) -> Dict[str, Any]:
        client = cls.get_client()
        return client.get_benchmark_status(benchmark_id)
    
    @classmethod
    def get_result(cls, benchmark_id: str) -> Dict[str, Any]:
        client = cls.get_client()
        return client.get_benchmark_result(benchmark_id)
    
    @classmethod
    def download_file(cls, benchmark_id: str, file_type: str) -> bytes:
        client = cls.get_client()
        return client.download_benchmark_file(benchmark_id, file_type)
    
    @classmethod
    def generate_queries(cls, databases: List[str], n_questions: int,
                         config: Dict[str, Any], models_infos: Dict[str, Any]) -> Dict[str, Any]:
        client = cls.get_client()
        return client.generate_queries(
            databases=databases,
            n_questions=n_questions,
            config=config,
            models_infos=models_infos
        )
    
    @classmethod
    def wait_for_completion(cls, benchmark_id: str, poll_interval: float = 2.0,
                           progress_callback: Optional[callable] = None,
                           max_wait: float = 3600.0) -> Dict[str, Any]:
        start_time = time.time()
        
        while True:
            status = cls.get_status(benchmark_id)
            current_status = status.get('status', 'unknown')
            
            if progress_callback:
                progress_callback(status)
            
            if current_status == 'completed':
                return cls.get_result(benchmark_id)
            
            if current_status == 'error':
                raise Exception(status.get('error', 'Unknown error during benchmark'))
            
            if time.time() - start_time > max_wait:
                raise TimeoutError(f"Benchmark did not complete within {max_wait} seconds")
            
            time.sleep(poll_interval)
    
    @classmethod
    def run_benchmark_sync(cls, rag_names: List[str], databases: List[str],
                           queries_doc_name: str, config: Dict[str, Any],
                           models_infos: Dict[str, Any], benchmark_type: str = 'full_bench',
                           reset_index: bool = False, reset_preprocess: bool = False,
                           progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        benchmark_id = cls.start_benchmark(
            rag_names=rag_names,
            databases=databases,
            queries_doc_name=queries_doc_name,
            config=config,
            models_infos=models_infos,
            benchmark_type=benchmark_type,
            reset_index=reset_index,
            reset_preprocess=reset_preprocess
        )
        
        if not benchmark_id:
            raise Exception("Failed to start benchmark")
        
        return cls.wait_for_completion(benchmark_id, progress_callback=progress_callback)
