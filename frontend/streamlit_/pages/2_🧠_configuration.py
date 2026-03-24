import streamlit as st
from urllib.parse import urlparse
from streamlit_.services import ConfigService
from streamlit_.api_client.exceptions import APIError
from streamlit_.utils.chat_funcs import get_chat_agent
from streamlit_.utils.params_func import get_custom_rags_name, modify_env

st.markdown('# Set Configuration')
host_llm = {'openai': 'OpenAI', 'mistral': 'Mistral'}

models_infos = st.session_state.get('models_infos', {})
if not models_infos and not st.session_state.get('backend_connected', True):
    st.warning('⚠️ Backend not connected and no models available')
    st.stop()


def set_false():
    for i in st.session_state['benchmark']['rags'].keys():
        st.session_state['benchmark']['rags'][i] = False


providers = st.session_state.get('providers_infos', {})
host_dict = {
    'ollama': {'url': providers.get('ollama', {}).get('url', ''), 'type': 'ollama', 'api_key': providers.get('ollama', {}).get('api_key', '')},
    'openai': {'url': providers.get('openai', {}).get('url', ''), 'type': 'openai', 'api_key': providers.get('openai', {}).get('api_key', '')}
}

st.subheader('📌 API/Models Configuration')
models_infos = st.session_state.get('models_infos', {})
model_names = list(models_infos.keys())
model_names.append('➕ Add a new model')
selected_model_name = st.selectbox('Select a model', model_names, key='selected_model_name')
if selected_model_name != '➕ Add a new model':
        selected_model = models_infos.get(selected_model_name, {})
        if st.session_state.get('selected_model_prev') != selected_model_name:
            st.session_state['edit_model_type'] = selected_model.get('type', 'llm')
            st.session_state['edit_model_api_key'] = selected_model.get('api_key', '')
            st.session_state['edit_model_url'] = selected_model.get('url', '')
            st.session_state['selected_model_prev'] = selected_model_name
if selected_model_name == '➕ Add a new model':
        new_model_name = st.text_input('Model name', key='new_model_name')
        new_model_url = st.text_input('Model URL', key='new_model_url')
        new_model_api_key = st.text_input('API Key', type='password', key='new_model_api_key')
        new_model_type = st.selectbox('Model type', ['llm', 'reranker', 'embedding'])
        new_model_provider = st.selectbox('Provider', ['openai', 'anthropic', 'mistral', 'cohere', 'custom (openaiSDK-compatible)'], key='new_model_provider')
        if st.button('Add model'):
            model_name_val = st.session_state.get('new_model_name', '')
            if model_name_val:
                if model_name_val in models_infos:
                    st.warning('This model already exists!')
                else:
                    models_infos[model_name_val] = {
                        'url': st.session_state.get('new_model_url', ''),
                        'api_key': st.session_state.get('new_model_api_key', ''),
                        'type': new_model_type,
                        'provider': st.session_state.get('new_model_provider', 'openai')
                    }
                    ConfigService.update_models(models_infos)
                    st.session_state['models_infos'] = models_infos
                    st.success(f"Model '{model_name_val}' added ✅")
            else:
                st.warning('Please fill in all fields!')
else:
            selected_model = models_infos.get(selected_model_name, {})
            with st.container():
                st.markdown(f'###### Model: {selected_model_name}')
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    new_model_type = st.selectbox('Model type', ['llm', 'reranker', 'embedding'], index=['llm', 'reranker', 'embedding'].index(selected_model.get('type', 'llm')), key='edit_model_type')
                with col2:
                    new_api_key = st.text_input('API Key', value=selected_model.get('api_key', ''), type='password', key='edit_model_api_key')
                with col3:
                    provider_options = ['openai', 'anthropic', 'mistral', 'custom (openaiSDK-compatible)']
                    current_provider = selected_model.get('provider', 'openai')
                    if current_provider in ['custom']:
                        current_provider = 'custom (openaiSDK-compatible)'
                    if current_provider not in provider_options:
                        current_provider = provider_options[0]
                    new_provider = st.selectbox('Provider', provider_options, index=provider_options.index(current_provider) if current_provider in provider_options else 0, key='edit_model_provider')
            new_url = st.text_input('URL', value=selected_model.get('url', ''), key='edit_model_url')
            col_empty, col1_btn, col_empty1, col2_btn, col_empty2 = st.columns([0.5, 2, 0.5, 2, 0.5])
            with col1_btn:
                if st.button('💾 Save changes', use_container_width=True):
                    models_infos[selected_model_name]['url'] = st.session_state.get('edit_model_url', selected_model.get('url', ''))
                    models_infos[selected_model_name]['api_key'] = st.session_state.get('edit_model_api_key', selected_model.get('api_key', ''))
                    models_infos[selected_model_name]['type'] = st.session_state.get('edit_model_type', selected_model.get('type', 'llm'))
                    models_infos[selected_model_name]['provider'] = st.session_state.get('edit_model_provider', selected_model.get('provider', 'openai'))
                    ConfigService.update_models(models_infos)
                    st.session_state['models_infos'] = models_infos
                    st.success('Changes saved ✅')
                    st.rerun()
            with col2_btn:
                if st.button('🗑️ Delete model', use_container_width=True):
                    del models_infos[selected_model_name]
                    ConfigService.update_models(models_infos)
                    st.session_state['models_infos'] = models_infos
                    st.success(f"Model '{selected_model_name}' deleted ✅")

st.markdown('<br><br>', unsafe_allow_html=True)
roles = {'Base LLM': 'model', 'Reranker': 'reranker_model', 'Model for image description': 'model_for_image', 'Embedding model': 'embedding_model'}
task_mapping = {'model': ['llm'], 'reranker_model': ['reranker', 'llm'], 'model_for_image': ['llm'], 'embedding_model': ['embedding']}

def sort_with_priority(model_name, config_key):
        model_type = models_infos[model_name]['type']
        priorities = task_mapping.get(config_key, [])
        priority_index = priorities.index(model_type) if model_type in priorities else len(priorities)
        return (priority_index, model_name)

config = {}
st.subheader('📌 Model role configuration')
for role_label, config_key in roles.items():
        if config_key not in st.session_state['config_server'].keys():
            st.session_state[config_key] = config.get(config_key, None)
        valid_tasks = task_mapping.get(config_key, [])
        filtered_models = [name for name, info in models_infos.items() if info.get('type') in valid_tasks]
        options = [None] + sorted(filtered_models)
        col1, col2 = st.columns([0.5, 2])
        with col1:
            st.markdown(f"**Model for {role_label}** {('✅' if st.session_state['config_server'].get(config_key) else '❌')}")
        with col2:
            config_value = st.session_state['config_server'].get(config_key)
            if config_value in options:
                index = options.index(config_value)
            else:
                index = 0
            selected_model = st.selectbox(label=f'Select {config_key}', options=options, index=index, format_func=lambda x: 'No model' if x is None else x, key=f'model_select_{config_key}', label_visibility='collapsed')
        st.session_state[config_key] = selected_model
        config[config_key] = selected_model
        st.session_state['config_server'][config_key] = selected_model

col_empty, col_save, col_empty1 = st.columns([1, 2, 1])
st.markdown('<br>', unsafe_allow_html=True)
with col_save:
        if st.button('💾 Save default models', use_container_width=True):
            ConfigService.update_config(st.session_state['config_server'])
            st.success('✅ Default models saved!')

st.markdown('<br>', unsafe_allow_html=True)
st.subheader('📊 Configured models availability')

col_btn, _ = st.columns([1, 4])
with col_btn:
    if st.button('🔄 Check availability', use_container_width=True):
        st.session_state['test_models'] = True

if st.session_state.get('test_models', False):
    try:
        with st.spinner('Testing models...'):
            # Save current model selections to server before testing
            model_keys = ['model', 'embedding_model', 'reranker_model', 'model_for_image']
            for key in model_keys:
                if key in st.session_state:
                    st.session_state['config_server'][key] = st.session_state[key]
            ConfigService.update_config(st.session_state['config_server'])
            # Now test the models
            test_results = ConfigService.test_models()
        st.session_state['models_test_results'] = test_results
        st.session_state['test_models'] = False
    except Exception as e:
        st.error(f"Error during testing: {e}")
        st.session_state['models_test_results'] = None

test_results = st.session_state.get('models_test_results', {})

if test_results:
    st.markdown('<br>', unsafe_allow_html=True)
    
    role_labels = {
        'model': 'Base LLM',
        'embedding_model': 'Embedding model',
        'reranker_model': 'Reranker',
        'model_for_image': 'Model for image description'
    }
    
    for key, label in role_labels.items():
        result = test_results.get(key)
        if result:
            status_icon = "🟢" if result.get('available', False) else "🔴"
            model_name = result.get('name', 'Aucun')
            
            st.markdown(f"**{status_icon} {label}**")
            
            if result.get('available', False):
                st.success(f"✅ {model_name} is available")
            else:
                error_msg = result.get('error', 'Unknown error')
                # Simplifier les messages d'erreur techniques
                simplified_error = error_msg

                # Erreurs de connexion
                if 'Network is unreachable' in error_msg or 'Failed to establish' in error_msg:
                    simplified_error = "Cannot connect to server (network unreachable)"
                elif 'Connection refused' in error_msg:
                    simplified_error = "Cannot connect to server (connection refused)"
                elif 'timeout' in error_msg.lower():
                    simplified_error = "Connection timeout"
                elif 'Max retries exceeded' in error_msg:
                    simplified_error = "Cannot connect to server (max retries exceeded)"

                # Erreurs de modèles
                elif 'non trouvé sur le serveur' in error_msg or 'not found on server' in error_msg.lower():
                    simplified_error = "Model not found on this server"
                elif 'non disponible' in error_msg or 'not available' in error_msg.lower():
                    simplified_error = "Model not available on this server"

                # Erreurs de endpoint
                elif 'Endpoint /v1/rerank non trouvé' in error_msg or 'not a reranking server' in error_msg.lower():
                    simplified_error = "Invalid URL (not a reranking server)"
                elif 'Réponse invalide du serveur rerank' in error_msg or 'invalid rerank response' in error_msg.lower():
                    simplified_error = "Invalid server response (not a reranking server)"

                # Garder le message original s'il est court, sinon le simplifier
                if len(error_msg) > 100:
                    display_error = simplified_error
                else:
                    display_error = error_msg

                st.error(f"❌ {model_name}: {display_error}")

st.markdown('<br><br>', unsafe_allow_html=True)
st.subheader('📌 Vectorbase Configuration')
new_elastic_url = st.text_input('URL Elasticsearch', value=st.session_state['config_server'].get('params_vectorbase', {}).get('url', ''), key='elastic_url')
col1, col2 = st.columns([2, 2])
with col1:
    new_elastic_auth = st.text_input('Auth', value=st.session_state['config_server'].get('params_vectorbase', {}).get('auth', ['', ''])[0], key='elastic_auth')
with col2:
    new_elastic_api_key = st.text_input('Clé API', type='password', value=st.session_state['config_server'].get('params_vectorbase', {}).get('auth', ['', ''])[1], key='elastic_api_key')
st.session_state['config_server']['params_vectorbase']['url'] = st.session_state.get('elastic_url', st.session_state['config_server'].get('params_vectorbase', {}).get('url', ''))
st.session_state['config_server']['params_vectorbase']['auth'][0] = st.session_state.get('elastic_auth', st.session_state['config_server'].get('params_vectorbase', {}).get('auth', ['', ''])[0])
st.session_state['config_server']['params_vectorbase']['auth'][1] = st.session_state.get('elastic_api_key', st.session_state['config_server'].get('params_vectorbase', {}).get('auth', ['', ''])[1])
if st.button('💾 Save elasticsearch params', use_container_width=True):
    ConfigService.update_config(st.session_state['config_server'])
    # Vérifier la connexion Elasticsearch avec la nouvelle configuration via le backend
    try:
        client = st.session_state.get('api_client')
        if client:
            result = client.check_elasticsearch_health()
            if result.get('status') == 'connected':
                st.success(f"✅ Elasticsearch connection successful! (version: {result.get('version', 'unknown')})")
                st.session_state['elasticsearch_connected'] = True
            else:
                st.error(f"❌ Elasticsearch connection failed: {result.get('error', 'Unknown error')}")
                st.session_state['elasticsearch_connected'] = False
        else:
            st.warning('⚠️ Backend not connected')
    except Exception as e:
        st.error(f'❌ Elasticsearch connection error: {e}')
        st.session_state['elasticsearch_connected'] = False
st.markdown('<br><br>', unsafe_allow_html=True)

data_preparation = {'pdf_text_extraction': 'PDF text extraction', 'md_without_images': 'PDF conversion into markdown'}
selected_data_prep = st.selectbox(label='**Choose data preparation method:**', options=list(data_preparation.keys()), format_func=lambda x: data_preparation[x], on_change=set_false, index=0, key='data_prep')


def reset_retrieval():
    st.session_state['ret'] = 'embeddings'

if 'ret' not in st.session_state:
    st.session_state['ret'] = st.session_state['config_server'].get('type_retrieval', 'embeddings')
if st.session_state['config_server'].get('params_vectorbase', {}).get('backend') == 'milvus':
    retrieval_methods = {'embeddings': 'Embeddings'}
    st.selectbox('**Choose retrieval method:**', retrieval_methods.keys(), format_func=lambda x: retrieval_methods[x], key='ret')
elif st.session_state['config_server'].get('params_vectorbase', {}).get('backend') == 'elasticsearch':
    retrieval_methods = {'embeddings': 'Embeddings', 'bm25': 'BM25', 'hybrid': 'Hybrid'}
    st.selectbox('**Choose retrieval method:**', retrieval_methods.keys(), format_func=lambda x: retrieval_methods[x], index=list(retrieval_methods.keys()).index(st.session_state['ret']), key='ret')
else:
    retrieval_methods = {'embeddings': 'Embeddings', 'bm25': 'BM25', 'hybrid': 'Hybrid'}
    st.selectbox('**Choose retrieval method**', retrieval_methods.keys(), format_func=lambda x: retrieval_methods[x], index=list(retrieval_methods.keys()).index(st.session_state['ret']), key='ret')

languages = ['FR', 'EN']
if 'lang' not in st.session_state:
    st.session_state['lang'] = st.session_state['config_server'].get('language', 'FR')
st.selectbox('**Choose RAG language:**', languages, index=languages.index(st.session_state['lang']), key='lang')
st.session_state['config_server']['language'] = st.session_state['lang']

splitter_dic = {'Semantic_TextSplitter': 'Semantic Splitting', 'Recursive_TextSplitter': 'Recursive Splitting', 'TextSplitter': 'Length Splitting'}
if 'split' not in st.session_state:
    st.session_state['split'] = st.session_state['config_server'].get('TextSplitter', 'TextSplitter')
st.selectbox('**Choose TextSplitter:**', splitter_dic.keys(), help='- **Semantic Splitting** : Spots semantic similarities between sentences and chunks accordingly \n\n                                - **Recursive Splitting** : Divides text into smaller segments in a hierarchical and iterative manner, using a series of separators to preserve the structure and context of the text. \n\n                                - **Length Splitting** : Splits text into chunks of fixed size\n                                ', format_func=lambda x: splitter_dic[x], index=list(splitter_dic.keys()).index(st.session_state['split']), key='split')

if 'reformulate' not in st.session_state:
    st.session_state['reformulate'] = st.session_state['config_server'].get('reformulate_query', False)
st.toggle('**Query reformulation ?**', help='Wether your query is reformulated by an LLM *before* being sent to the RAG', value=st.session_state['reformulate'], key='reformulate')

if 'chunk' not in st.session_state:
    st.session_state['chunk'] = st.session_state['config_server'].get('nb_chunks', 5)
st.slider(label='**Choose number of chunks to retrieve per query:**', min_value=0, max_value=500, step=5, value=st.session_state['chunk'], help='The higher the number of value, the better the results of the RAG agent will be.\n                                                           However a number of chunk too large might slow down the answer time and increase costs', key='chunk')
st.session_state['config_server']['nb_chunks'] = st.session_state['chunk']

if st.button('Save Configuration', type='primary', use_container_width=True):
    # Save locally modified parameters before calling server
    local_params_to_preserve = {
        'TextSplitter': st.session_state['split'],
        'reformulate_query': st.session_state['reformulate'],
        'type_retrieval': st.session_state['ret'],
        'data_preprocessing': st.session_state['data_prep'],
        'nb_chunks': st.session_state['chunk'],
        'language': st.session_state['lang'],
        # Models by role
        'model': st.session_state.get('model'),
        'embedding_model': st.session_state.get('embedding_model'),
        'reranker_model': st.session_state.get('reranker_model'),
        'model_for_image': st.session_state.get('model_for_image'),
        # Elasticsearch parameters
        'params_vectorbase': st.session_state['config_server'].get('params_vectorbase', {}),
    }

    # Mettre à jour la config avec les valeurs locales
    for key, value in local_params_to_preserve.items():
        if value is not None:  # Ne pas écraser avec None
            st.session_state['config_server'][key] = value

    try:
        result = ConfigService.change_server_config(rag_name=None)
        server_config = result.get('config', {})
        # Restaurer les paramètres locaux après l'appel au serveur
        for key, value in local_params_to_preserve.items():
            if value is not None:
                server_config[key] = value
        # Fusionner seulement les clés qui ne sont pas dans nos paramètres locaux
        for key, value in server_config.items():
            if key not in local_params_to_preserve or local_params_to_preserve.get(key) is None:
                st.session_state['config_server'][key] = value
        st.success('Configuration saved ✅')
    except APIError as e:
        st.warning(f"API error: {e}")

    ConfigService.update_config(st.session_state['config_server'])
    
    st.session_state['custom_rags'] = get_custom_rags_name()
    rag_method = st.session_state['rag_name']
    rag_agent = get_chat_agent(rag_method=rag_method, databases_name=[])
    st.session_state['success'] = True
    st.session_state['rag'] = rag_agent
    st.session_state['rag_name'] = rag_method
