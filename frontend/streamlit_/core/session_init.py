import pandas as pd
from streamlit_.utils.params_func import get_custom_rags_name, get_merge_rags_name
from streamlit_.api_client import APIClient
from streamlit_.api_client.exceptions import APIError
from streamlit_.core.config import API_BASE_URL


def init_session_state(st):
    _init_api_client(st)
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


def _init_api_client(st):
    if 'api_client' not in st.session_state:
        st.session_state['api_client'] = APIClient(base_url=API_BASE_URL)
    if 'api_session_id' not in st.session_state:
        st.session_state['api_session_id'] = None


def _init_config(st):
    if 'config_server' not in st.session_state:
        try:
            response = st.session_state['api_client'].get_config()
            st.session_state['config_server'] = response.get('config', {})
            st.session_state['config_server']['local_params'] = response.get('local_params', {
                'forced_system_prompt': False,
                'generation_system_prompt_name': 'default'
            })
        except APIError as e:
            st.error(f'Error loading config from API: {e}')
            st.session_state['config_server'] = {}


def _init_rag_name(st):
    if 'rag_name' not in st.session_state:
        st.session_state['rag_name'] = 'naive'
    if 'selected_rag_method' not in st.session_state:
        st.session_state['selected_rag_method'] = 'naive'


def _init_providers(st):
    if 'providers_infos' not in st.session_state:
        try:
            st.session_state['providers_infos'] = st.session_state['api_client'].get_providers()
        except APIError as e:
            st.error(f'Error loading providers from API: {e}')
            st.session_state['providers_infos'] = {}


def _init_api_key(st):
    if 'api_key' not in st.session_state:
        provider_default_mode = st.session_state['config_server'].get('default_mode_provider', 'ollama')
        providers = st.session_state.get('providers_infos', {})
        if provider_default_mode in providers:
            st.session_state['api_key'] = providers[provider_default_mode].get('api_key')


def _init_models(st):
    if 'models_infos' not in st.session_state:
        try:
            st.session_state['models_infos'] = st.session_state['api_client'].get_models()
        except APIError as e:
            st.error(f'Error loading models from API: {e}')
            st.session_state['models_infos'] = {}


def _init_all_rags(st):
    if 'all_rags' not in st.session_state:
        try:
            st.session_state['all_rags'] = st.session_state['api_client'].get_all_rags()
        except APIError as e:
            st.error(f'Error loading RAG methods from API: {e}')
            st.session_state['all_rags'] = {}


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


def _init_all_databases(st):
    if 'all_databases' not in st.session_state:
        try:
            databases = st.session_state['api_client'].list_databases()
            all_db = [db['name'] for db in databases]
            st.session_state['all_databases'] = all_db
            if 'chat_database_name' not in st.session_state:
                st.session_state['chat_database_name'] = all_db[0] if len(all_db) > 0 else None
            if 'benchmark_database' not in st.session_state:
                st.session_state['benchmark_database'] = []
        except APIError as e:
            st.error(f'Error loading databases from API: {e}')
            st.session_state['all_databases'] = []
            st.session_state['chat_database_name'] = None
            st.session_state['benchmark_database'] = []
