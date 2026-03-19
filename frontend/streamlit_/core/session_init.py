import pandas as pd
from datetime import datetime
from streamlit_.utils.params_func import get_custom_rags_name, get_merge_rags_name
from streamlit_.api_client import APIClient
from streamlit_.api_client.exceptions import APIError
from streamlit_.core.config import API_BASE_URL


def init_session_state(st):
    _init_api_client(st)
    _init_connection_status(st)
    
    is_backend_up = _check_backend_connection(st)
    just_came_up = st.session_state.get('backend_just_came_up', False)
    
    if is_backend_up and just_came_up:
        _init_config(st, force=True)
        _init_providers(st, force=True)
        _init_models(st, force=True)
        _init_all_rags(st, force=True)
        _init_all_databases(st, force=True)
        _init_rag_name(st)
        _init_api_key(st)
        _init_benchmark(st)
        _init_custom_rags(st)
        _init_databases(st)
        _init_merge_rags(st)
        _init_success(st)
        return
    
    if not is_backend_up:
        _init_defaults(st)
        return
    
    _init_config(st)
    _init_rag_name(st)
    _init_providers(st)
    _init_api_key(st)
    _init_models(st)
    _init_all_rags(st)
    _init_benchmark(st)
    _init_custom_rags(st)
    _init_databases(st)
    _init_merge_rags(st)
    _init_success(st)
    _init_all_databases(st)


def _check_backend_connection(st) -> bool:
    CACHE_DURATION = 30
    
    current_time = datetime.now()
    
    if st.session_state.get('force_backend_check', False):
        st.session_state['last_backend_check'] = datetime.min
        st.session_state['force_backend_check'] = False
    
    if 'last_backend_check' in st.session_state:
        elapsed = (current_time - st.session_state['last_backend_check']).total_seconds()
        if elapsed < CACHE_DURATION:
            return st.session_state.get('backend_connected', False)
    
    was_connected = st.session_state.get('backend_connected', False)
    
    try:
        is_connected = st.session_state['api_client'].health_check()
        st.session_state['backend_connected'] = is_connected
        st.session_state['last_backend_check'] = current_time
        
        if not was_connected and is_connected:
            st.session_state['backend_just_came_up'] = True
        else:
            st.session_state['backend_just_came_up'] = False
        
        return is_connected
    except:
        st.session_state['backend_connected'] = False
        st.session_state['last_backend_check'] = current_time
        st.session_state['backend_just_came_up'] = False
        return False


def _init_defaults(st):
    if 'config_server' not in st.session_state:
        st.session_state['config_server'] = {
            'local_params': {
                'forced_system_prompt': False,
                'generation_system_prompt_name': 'default'
            },
            'all_system_prompt': {'default': 'Default prompt'},
            'default_mode_provider': 'ollama',
            'params_vectorbase': {
                'url': 'http://localhost:9200',
                'auth': ['elastic', ''],
                'backend': 'elasticsearch'
            },
            'type_retrieval': 'embeddings',
            'data_preprocessing': 'pdf_text_extraction',
            'nb_chunks_reranker': 100,
            'nb_chunks': 5,
            'chunk_length': 512,
            'model': None,
            'reranker_model': None,
            'model_for_image': None,
            'embedding_model': None,
            'TextSplitter': 'TextSplitter',
            'language': 'FR',
            'reformulate_query': False
        }
    if 'rag_name' not in st.session_state:
        st.session_state['rag_name'] = 'naive'
    if 'selected_rag_method' not in st.session_state:
        st.session_state['selected_rag_method'] = 'naive'
    if 'providers_infos' not in st.session_state:
        st.session_state['providers_infos'] = {}
    if 'models_infos' not in st.session_state:
        st.session_state['models_infos'] = {}
    if 'all_rags' not in st.session_state:
        st.session_state['all_rags'] = {}
    if 'benchmark' not in st.session_state:
        st.session_state['benchmark'] = {
            'rags': {},
            'queries': pd.DataFrame(data={'query': [], 'answer': []}),
            'load': False
        }
    if 'custom_rags' not in st.session_state:
        st.session_state['custom_rags'] = []
    if 'databases' not in st.session_state:
        st.session_state['databases'] = {}
    if 'merge_rags' not in st.session_state:
        st.session_state['merge_rags'] = []
    if 'rags_to_merge' not in st.session_state:
        st.session_state['rags_to_merge'] = {
            'rags': {},
            'queries': pd.DataFrame(data={'query': [], 'answer': []}),
            'load': False
        }
    if 'success' not in st.session_state:
        st.session_state['success'] = False
    if 'all_databases' not in st.session_state:
        st.session_state['all_databases'] = []
        st.session_state['chat_database_name'] = None
        st.session_state['benchmark_database'] = []


def _init_api_client(st):
    if 'api_client' not in st.session_state:
        st.session_state['api_client'] = APIClient(base_url=API_BASE_URL)
    if 'api_session_id' not in st.session_state:
        st.session_state['api_session_id'] = None


def _init_config(st, force=False):
    CACHE_DURATION = 60
    current_time = datetime.now()
    
    if 'config_server_last_update' not in st.session_state:
        st.session_state['config_server_last_update'] = datetime.min
    
    elapsed = (current_time - st.session_state['config_server_last_update']).total_seconds()
    
    if force or elapsed >= CACHE_DURATION or 'config_server' not in st.session_state:
        if st.session_state.get('backend_connected', False):
            try:
                response = st.session_state['api_client'].get_config()
                st.session_state['config_server'] = response.get('config', {})
                st.session_state['config_server']['local_params'] = response.get('local_params', {
                    'forced_system_prompt': False,
                    'generation_system_prompt_name': 'default'
                })
                st.session_state['config_server_last_update'] = current_time
            except APIError:
                pass


def _init_rag_name(st):
    if 'rag_name' not in st.session_state:
        st.session_state['rag_name'] = 'naive'
    if 'selected_rag_method' not in st.session_state:
        st.session_state['selected_rag_method'] = 'naive'


def _init_providers(st, force=False):
    CACHE_DURATION = 60
    current_time = datetime.now()
    
    if 'providers_infos_last_update' not in st.session_state:
        st.session_state['providers_infos_last_update'] = datetime.min
    
    elapsed = (current_time - st.session_state['providers_infos_last_update']).total_seconds()
    
    if force or elapsed >= CACHE_DURATION or 'providers_infos' not in st.session_state:
        if st.session_state.get('backend_connected', False):
            try:
                st.session_state['providers_infos'] = st.session_state['api_client'].get_providers()
                st.session_state['providers_infos_last_update'] = current_time
            except APIError:
                pass


def _init_api_key(st):
    if 'api_key' not in st.session_state:
        provider_default_mode = st.session_state['config_server'].get('default_mode_provider', 'ollama')
        providers = st.session_state.get('providers_infos', {})
        if provider_default_mode in providers:
            st.session_state['api_key'] = providers[provider_default_mode].get('api_key')


def _init_models(st, force=False):
    CACHE_DURATION = 60
    current_time = datetime.now()
    
    if 'models_infos_last_update' not in st.session_state:
        st.session_state['models_infos_last_update'] = datetime.min
    
    elapsed = (current_time - st.session_state['models_infos_last_update']).total_seconds()
    
    if force or elapsed >= CACHE_DURATION or 'models_infos' not in st.session_state:
        if st.session_state.get('backend_connected', False):
            try:
                st.session_state['models_infos'] = st.session_state['api_client'].get_models()
                st.session_state['models_infos_last_update'] = current_time
            except APIError:
                pass


def _init_all_rags(st, force=False):
    CACHE_DURATION = 60
    current_time = datetime.now()
    
    if 'all_rags_last_update' not in st.session_state:
        st.session_state['all_rags_last_update'] = datetime.min
    
    elapsed = (current_time - st.session_state['all_rags_last_update']).total_seconds()
    
    if force or elapsed >= CACHE_DURATION or 'all_rags' not in st.session_state:
        if st.session_state.get('backend_connected', False):
            try:
                st.session_state['all_rags'] = st.session_state['api_client'].get_all_rags()
                st.session_state['all_rags_last_update'] = current_time
            except APIError:
                pass


def _init_benchmark(st):
    if 'benchmark' not in st.session_state:
        st.session_state['benchmark'] = {}
        list_rags = list(st.session_state['all_rags'].keys())
        list_rags = set(list_rags)
        st.session_state['benchmark']['rags'] = dict(zip(list_rags, [False for _ in range(len(list_rags))]))
        st.session_state['benchmark']['queries'] = pd.DataFrame(data={'query': [], 'answer': []})
        st.session_state['benchmark']['load'] = False


def _init_custom_rags(st):
    st.session_state['custom_rags'] = get_custom_rags_name()
    if '.gitkeep' in st.session_state['custom_rags']:
        st.session_state['custom_rags'].remove('.gitkeep')


def _init_databases(st):
    if 'databases' not in st.session_state:
        st.session_state['databases'] = {}


def _init_merge_rags(st):
    if 'merge_rags' not in st.session_state:
        st.session_state['merge_rags'] = get_merge_rags_name()
    if 'rags_to_merge' not in st.session_state:
        st.session_state['rags_to_merge'] = {}
        list_rags = list(st.session_state['all_rags'].keys())
        list_rags = set(list_rags)
        st.session_state['rags_to_merge']['rags'] = dict(zip(list_rags, [False for _ in range(len(list_rags))]))
        st.session_state['rags_to_merge']['queries'] = pd.DataFrame(data={'query': [], 'answer': []})
        st.session_state['rags_to_merge']['load'] = False


def _init_success(st):
    if 'success' not in st.session_state:
        st.session_state['success'] = False


def _init_all_databases(st, force=False):
    CACHE_DURATION = 60
    current_time = datetime.now()
    
    if 'all_databases_last_update' not in st.session_state:
        st.session_state['all_databases_last_update'] = datetime.min
    
    elapsed = (current_time - st.session_state['all_databases_last_update']).total_seconds()
    
    if force or elapsed >= CACHE_DURATION or 'all_databases' not in st.session_state:
        if st.session_state.get('backend_connected', False):
            try:
                databases = st.session_state['api_client'].list_databases()
                all_db = [db['name'] for db in databases]
                st.session_state['all_databases'] = all_db
                if 'chat_database_name' not in st.session_state:
                    st.session_state['chat_database_name'] = all_db[0] if len(all_db) > 0 else None
                if 'benchmark_database' not in st.session_state:
                    st.session_state['benchmark_database'] = []
                st.session_state['all_databases_last_update'] = current_time
            except APIError:
                st.session_state['all_databases'] = []
                st.session_state['chat_database_name'] = None
                st.session_state['benchmark_database'] = []


def _check_elasticsearch_connection(st) -> bool:
    """Check Elasticsearch connection using the backend API"""
    try:
        client = st.session_state.get('api_client')
        if not client:
            return False

        result = client.check_elasticsearch_health()
        return result.get('status') == 'connected'
    except Exception as e:
        return False


def _init_connection_status(st):
    if 'backend_connected' not in st.session_state:
        st.session_state['backend_connected'] = True
    if 'elasticsearch_connected' not in st.session_state:
        st.session_state['elasticsearch_connected'] = True
    if 'last_health_check' not in st.session_state:
        st.session_state['last_health_check'] = datetime.min
    if 'last_es_check' not in st.session_state:
        st.session_state['last_es_check'] = datetime.min
