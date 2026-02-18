from typing import Dict, Any, Optional, List, Union
import requests

from .exceptions import APIError, SessionNotFoundError, AgentNotFoundError


class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self._session_id: Optional[str] = None
    
    @property
    def session_id(self) -> Optional[str]:
        return self._session_id
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code == 404:
                data = response.json()
                if 'session' in data.get('detail', '').lower():
                    raise SessionNotFoundError(self._session_id or 'unknown')
                if 'agent' in data.get('detail', '').lower():
                    raise AgentNotFoundError(self._session_id or 'unknown')
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise APIError(str(e))
    
    def health_check(self) -> bool:
        try:
            response = self._request('GET', '/health')
            return response.get('status') == 'healthy'
        except APIError:
            return False
    
    def create_session(self) -> str:
        response = self._request('POST', '/api/session')
        self._session_id = response['session_id']
        return self._session_id or ''
    
    def get_session_info(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid = session_id or self._session_id
        if not sid:
            raise SessionNotFoundError('No session')
        return self._request('GET', f'/api/session/{sid}')
    
    def delete_session(self, session_id: Optional[str] = None) -> bool:
        sid = session_id or self._session_id
        if not sid:
            return False
        response = self._request('DELETE', f'/api/session/{sid}')
        if sid == self._session_id:
            self._session_id = None
        return response.get('status') == 'deleted'
    
    def list_rag_methods(self) -> Any:
        response = self._request('GET', '/api/rag/methods')
        return response
    
    def create_agent(self, rag_method: str, config: Dict[str, Any], 
                     models_infos: Dict[str, Any], databases: List[str],
                     session_id: Optional[str] = None) -> Dict[str, Any]:
        sid = session_id or self._session_id
        if not sid:
            sid = self.create_session()
        
        response = self._request('POST', '/api/rag/create', json={
            'session_id': sid,
            'rag_method': rag_method,
            'config': config,
            'models_infos': models_infos,
            'databases': databases
        })
        return response
    
    def run_indexation(self, reset_index: bool = False, reset_preprocess: bool = False,
                       session_id: Optional[str] = None) -> Dict[str, Any]:
        sid = session_id or self._session_id
        if not sid:
            raise SessionNotFoundError('No session')
        
        return self._request('POST', '/api/rag/index', json={
            'session_id': sid,
            'reset_index': reset_index,
            'reset_preprocess': reset_preprocess
        })
    
    def generate_answer(self, query: str, nb_chunks: int = 5,
                        options_generation: Optional[Dict[str, Any]] = None,
                        session_id: Optional[str] = None) -> Dict[str, Any]:
        sid = session_id or self._session_id
        if not sid:
            raise SessionNotFoundError('No session')
        
        payload = {
            'session_id': sid,
            'query': query,
            'nb_chunks': nb_chunks
        }
        if options_generation:
            payload['options_generation'] = options_generation
        
        return self._request('POST', '/api/rag/generate', json=payload)
    
    def get_agent_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid = session_id or self._session_id
        if not sid:
            raise SessionNotFoundError('No session')
        return self._request('GET', f'/api/rag/status/{sid}')
    
    def generate_rag_names(self, rag_name: str, config: Dict[str, Any], 
                           additional_name: str = '') -> Dict[str, Any]:
        return self._request('POST', '/api/rag/generate-names', json={
            'rag_name': rag_name,
            'config': config,
            'additional_name': additional_name
        })
    
    def list_elasticsearch_indices(self, prefix: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if prefix:
            params['prefix'] = prefix
        return self._request('GET', '/api/rag/elasticsearch/indices', params=params)
    
    def delete_elasticsearch_index(self, index_name: str) -> Dict[str, Any]:
        return self._request('DELETE', f'/api/rag/elasticsearch/indices/{index_name}')
    
    def delete_elasticsearch_indices_by_prefix(self, prefix: str) -> Dict[str, Any]:
        return self._request('DELETE', '/api/rag/elasticsearch/indices/batch', params={'prefix': prefix})
    
    def list_custom_rags(self) -> Dict[str, Any]:
        return self._request('GET', '/api/rag/custom')
    
    def create_custom_rag(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        return self._request('POST', '/api/rag/custom', json={
            'name': name,
            'config': config
        })
    
    def delete_custom_rag(self, name: str) -> Dict[str, Any]:
        return self._request('DELETE', f'/api/rag/custom/{name}')
    
    def get_custom_rag(self, name: str) -> Dict[str, Any]:
        return self._request('GET', f'/api/rag/custom/{name}')
    
    def list_merge_rags(self) -> Dict[str, Any]:
        return self._request('GET', '/api/rag/merge')
    
    def get_merge_rag(self, name: str) -> Dict[str, Any]:
        return self._request('GET', f'/api/rag/merge/{name}')
    
    def create_merge_rag(self, name: str, rag_list: List[str], 
                         rag_config_list: List[Dict[str, Any]], 
                         config: Dict[str, Any]) -> Dict[str, Any]:
        return self._request('POST', '/api/rag/merge', json={
            'name': name,
            'rag_list': rag_list,
            'rag_config_list': rag_config_list,
            'config': config
        })
    
    def delete_merge_rag(self, name: str) -> Dict[str, Any]:
        return self._request('DELETE', f'/api/rag/merge/{name}')
    
    def get_config(self) -> Dict[str, Any]:
        return self._request('GET', '/api/config')
    
    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return self._request('PUT', '/api/config', json={'config': config})
    
    def update_local_params(self, forced_system_prompt: bool = False,
                            generation_system_prompt_name: str = 'default') -> Dict[str, Any]:
        return self._request('PUT', '/api/config/local-params', json={
            'forced_system_prompt': forced_system_prompt,
            'generation_system_prompt_name': generation_system_prompt_name
        })
    
    def reset_local_params(self) -> Dict[str, Any]:
        return self._request('POST', '/api/config/local-params/reset')
    
    def get_system_info(self) -> Dict[str, Any]:
        return self._request('GET', '/api/config/system')
    
    def change_server_config(self, rag_name: Optional[str] = None, 
                             mode: str = 'Simple') -> Dict[str, Any]:
        return self._request('PUT', '/api/config/change-server', json={
            'rag_name': rag_name,
            'mode': mode
        })
    
    def get_models(self) -> Dict[str, Any]:
        return self._request('GET', '/api/config/models')
    
    def update_models(self, models: Dict[str, Any]) -> Dict[str, Any]:
        return self._request('PUT', '/api/config/models', json={'models': models})
    
    def get_providers(self) -> Dict[str, Any]:
        return self._request('GET', '/api/config/providers')
    
    def update_providers(self, providers: Dict[str, Any]) -> Dict[str, Any]:
        return self._request('PUT', '/api/config/providers', json={'providers': providers})
    
    def get_all_rags(self) -> Dict[str, Any]:
        return self._request('GET', '/api/config/all-rags')
    
    def update_all_rags(self, all_rags: Dict[str, Any]) -> Dict[str, Any]:
        return self._request('PUT', '/api/config/all-rags', json={'config': all_rags})
    
    def list_databases(self) -> List[Dict[str, Any]]:
        response = self._request('GET', '/api/databases')
        return response.get('databases', [])
    
    def create_database(self, name: str, embedding_model: str = '') -> Dict[str, Any]:
        return self._request('POST', '/api/databases', json={
            'name': name,
            'embedding_model': embedding_model
        })
    
    def delete_database(self, name: str) -> Dict[str, Any]:
        return self._request('DELETE', f'/api/databases/{name}')
    
    def get_database_documents(self, name: str) -> List[str]:
        response = self._request('GET', f'/api/databases/{name}/documents')
        return response.get('documents', [])
    
    def upload_document(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            return self._request('POST', '/api/documents/upload', files=files)
    
    def process_document(self, file_name: str, file_data: bytes) -> Dict[str, Any]:
        files = {'file': (file_name, file_data)}
        url = f"{self.base_url}/api/documents/process"
        response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()
    
    def get_document_content(self, document_id: str) -> str:
        response = self._request('GET', f'/api/documents/{document_id}/content')
        return response.get('content', '')
    
    def list_benchmark_reports(self) -> List[Dict[str, Any]]:
        response = self._request('GET', '/api/benchmark/reports')
        return response.get('reports', [])
    
    def get_benchmark_report(self, report_id: str) -> Dict[str, Any]:
        return self._request('GET', f'/api/benchmark/report/{report_id}')
    
    def delete_benchmark_report(self, report_id: str) -> Dict[str, Any]:
        return self._request('DELETE', f'/api/benchmark/report/{report_id}')
    
    def run_benchmark(self, session_ids: List[str], rag_names: List[str],
                      databases: List[str], queries_doc_name: str,
                      config: Dict[str, Any], models_infos: Dict[str, Any],
                      benchmark_type: str = 'all') -> Dict[str, Any]:
        return self._request('POST', '/api/benchmark/run', json={
            'session_ids': session_ids,
            'rag_names': rag_names,
            'databases': databases,
            'queries_doc_name': queries_doc_name,
            'config': config,
            'models_infos': models_infos,
            'benchmark_type': benchmark_type
        })
    
    def start_benchmark(self, rag_names: List[str], databases: List[str],
                        queries_doc_name: str, config: Dict[str, Any],
                        models_infos: Dict[str, Any], benchmark_type: str = 'full_bench',
                        reset_index: bool = False, reset_preprocess: bool = False) -> Dict[str, Any]:
        return self._request('POST', '/api/benchmark/start', json={
            'rag_names': rag_names,
            'databases': databases,
            'queries_doc_name': queries_doc_name,
            'config': config,
            'models_infos': models_infos,
            'benchmark_type': benchmark_type,
            'reset_index': reset_index,
            'reset_preprocess': reset_preprocess
        })
    
    def get_benchmark_status(self, benchmark_id: str) -> Dict[str, Any]:
        return self._request('GET', f'/api/benchmark/{benchmark_id}/status')
    
    def get_benchmark_result(self, benchmark_id: str) -> Dict[str, Any]:
        return self._request('GET', f'/api/benchmark/{benchmark_id}/result')
    
    def download_benchmark_file(self, benchmark_id: str, file_type: str) -> bytes:
        url = f"{self.base_url}/api/benchmark/{benchmark_id}/download/{file_type}"
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    
    def generate_queries(self, databases: List[str], n_questions: int,
                         config: Dict[str, Any], models_infos: Dict[str, Any]) -> Dict[str, Any]:
        return self._request('POST', '/api/benchmark/generate-queries', json={
            'databases': databases,
            'n_questions': n_questions,
            'config': config,
            'models_infos': models_infos
        })
    
    def list_query_files(self) -> List[Dict[str, Any]]:
        response = self._request('GET', '/api/queries')
        return response.get('queries', [])
    
    def upload_query_file(self, file_name: str, file_data: bytes) -> Dict[str, Any]:
        files = {'file': (file_name, file_data)}
        url = f"{self.base_url}/api/queries/upload"
        response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()
    
    def get_query_file(self, filename: str) -> Dict[str, Any]:
        return self._request('GET', f'/api/queries/{filename}')
    
    def delete_query_file(self, filename: str) -> Dict[str, Any]:
        return self._request('DELETE', f'/api/queries/{filename}')
    
    def list_storage_files(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {}
        if prefix:
            params['prefix'] = prefix
        response = self._request('GET', '/api/storage', params=params)
        return response.get('files', [])
    
    def delete_storage_file(self, filename: str) -> Dict[str, Any]:
        return self._request('DELETE', f'/api/storage/{filename}')
    
    def delete_storage_by_prefix(self, prefix: str) -> Dict[str, Any]:
        return self._request('DELETE', f'/api/storage/by-prefix/{prefix}')
    
    def upload_database_document(self, db_name: str, file_name: str, file_data: bytes) -> Dict[str, Any]:
        files = {'file': (file_name, file_data)}
        url = f"{self.base_url}/api/databases/{db_name}/documents/upload"
        response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()
    
    def delete_database_document(self, db_name: str, document_name: str) -> Dict[str, Any]:
        return self._request('DELETE', f'/api/databases/{db_name}/documents/{document_name}')
    
    def get_database_metadatas(self, db_name: str) -> Dict[str, Any]:
        response = self._request('GET', f'/api/databases/{db_name}/metadatas')
        return response.get('metadatas', {})
    
    def update_database_metadatas(self, db_name: str, metadatas: Dict[str, Any]) -> Dict[str, Any]:
        return self._request('PUT', f'/api/databases/{db_name}/metadatas', json={'metadatas': metadatas})
