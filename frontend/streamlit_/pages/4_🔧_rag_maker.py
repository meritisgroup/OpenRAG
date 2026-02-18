import streamlit as st
import json
import re
import pandas as pd
from streamlit_.api_client import APIClient
from streamlit_.api_client.exceptions import APIError
from streamlit_.core.config import API_BASE_URL
from streamlit_.utils.params_func import get_possible_embeddings_model, get_default_embeddings_model, get_config_rag

_client = APIClient(API_BASE_URL)

try:
    all_rags_response = _client.get_all_rags()
    base_rags = dict(all_rags_response)
except APIError as e:
    st.error(f"Error loading RAG methods: {e}")
    base_rags = {}

if 'naive_chatbot' in base_rags:
    del base_rags['naive_chatbot']

config_new_rag = st.session_state['config_server'].copy()
st.markdown('# Customize your RAG:')
config_new_rag['base'] = st.selectbox(label='**Choose your base RAG**', options=base_rags.keys(), format_func=lambda x: base_rags[x])
vectorbase_dict = {'elasticsearch': {'url': st.session_state['config_server']['params_vectorbase']['url'], 'backend': 'elasticsearch', 'auth': [st.session_state['config_server']['params_vectorbase']['auth'][0], st.session_state['config_server']['params_vectorbase']['auth'][1]], 'batch': True}}
backend_vectorbase = {'elasticsearch': 'Elastic Search'}
config_new_rag['params_vectorbase'] = vectorbase_dict['elasticsearch']
retrieval_methods = {'embeddings': 'Embeddings', 'bm25': 'BM25', 'hybrid': 'Hybrid'}
type_retrieval = st.selectbox('**Choose retrieval method**', retrieval_methods.keys(), format_func=lambda x: retrieval_methods[x], index=list(retrieval_methods.keys()).index(st.session_state['config_server']['type_retrieval']))
config_new_rag['type_retrieval'] = type_retrieval
data_preparation = {'pdf_text_extraction': 'PDF text extraction', 'md_without_images': 'PDF conversion into markdown'}
selected_data_prep = st.selectbox(label='**Choose data preparation method:**', options=list(data_preparation.keys()), format_func=lambda x: data_preparation[x], index=list(data_preparation.keys()).index(st.session_state['config_server']['data_preprocessing']), key='data_prep')
config_new_rag['data_preprocessing'] = selected_data_prep
splitter_dic = {'Semantic_TextSplitter': 'Semantic Splitting', 'Recursive_TextSplitter': 'Recursive Splitting', 'TextSplitter': 'Length Splitting'}
config_new_rag['TextSplitter'] = st.selectbox(label='**Choose text splitter**', options=splitter_dic.keys(), format_func=lambda x: splitter_dic[x])
if config_new_rag['base'] == 'advanced_rag' or config_new_rag['base'] == 'agentic':
    pre_proccessor_dic = {'Contextual integration': 'Contextual', 'Global Sum up document integration': 'Global_sum_up', 'Extract metadata': 'Extractor_metadata'}
    selected_processors = []
    list_keys_processor = list(pre_proccessor_dic.keys())
    nb_processor = len(list_keys_processor)
    processor_per_column = nb_processor // 3 if nb_processor % 3 == 0 else nb_processor // 3 + 1
    st.write('**Choose Pre-Processor Chunks**')
    col1, col2, col3 = st.columns(3)
    with col1:
        for i in range(processor_per_column):
            checked = st.checkbox(list_keys_processor[i], key=f'preproc_{i}')
            if checked:
                selected_processors.append(pre_proccessor_dic[list_keys_processor[i]])
    with col2:
        for i in range(processor_per_column, 2 * processor_per_column):
            checked = st.checkbox(list_keys_processor[i], key=f'preproc_{i}')
            if checked:
                selected_processors.append(pre_proccessor_dic[list_keys_processor[i]])
    with col3:
        for i in range(2 * processor_per_column, nb_processor):
            checked = st.checkbox(list_keys_processor[i], key=f'preproc_{i}')
            if checked:
                selected_processors.append(pre_proccessor_dic[list_keys_processor[i]])
    config_new_rag['ProcessorChunks'] = selected_processors
    nb_reranker = st.slider(label='**Choose number of chunks after reranker**', min_value=0, max_value=500, step=5, value=st.session_state['config_server']['nb_chunks_reranker'], help='The higher the number of chunks, the better the RAG agent might perform. However, a number of chunks too large can slow down responses and increase costs.', key='chunk')
    config_new_rag['nb_chunks_reranker'] = nb_reranker
if st.session_state['mode_interface'] == 'Simple':
    possible_embeddings = get_possible_embeddings_model(provider=st.session_state['config_server']['default_mode_provider'])
    if possible_embeddings:
        possible_embeddings.insert(0, 'default')
    else:
        possible_embeddings = ['default']
    embeddings_dic = {m: m for m in possible_embeddings}
    config_new_rag['embedding_model'] = st.selectbox(label='**Choose embedding model**', options=embeddings_dic.keys(), format_func=lambda x: embeddings_dic[x])
    if config_new_rag['embedding_model'] != 'default':
        st.warning(f"⚠️ This RAG will be only available for '{st.session_state['config_server']['default_mode_provider']}' provider.")
    else:
        config_new_rag['embedding_model'] = possible_embeddings[1] if len(possible_embeddings) > 1 else 'default'
else:
    models_infos = st.session_state.get('models_infos', {})
    st.subheader('📌 Models configuration')
    llm_models = list(models_infos.keys())
    config_new_rag['model'] = st.selectbox(label='**Choose LLm model**', options=llm_models, format_func=lambda x: x)
    reranker_models = list(models_infos.keys())
    config_new_rag['reranker_model'] = st.selectbox(label='**Choose Reranker model**', options=reranker_models, format_func=lambda x: x)
    image_models = list(models_infos.keys())
    config_new_rag['model_for_image'] = st.selectbox(label='**Choose model for image description**', options=image_models, format_func=lambda x: x)
    embeddings_models = list(models_infos.keys())
    config_new_rag['embedding_model'] = st.selectbox(label='**Choose embedding model**', options=embeddings_models, format_func=lambda x: x)
    st.markdown('<br><br>', unsafe_allow_html=True)
config_new_rag['nb_chunks'] = st.slider(label='**Choose number of chunks to retrieve per query**', min_value=0, max_value=500, step=5, value=st.session_state['config_server']['nb_chunks'], help='The higher the number of value, the better the results of the RAG agent will be.\n                                                           However a number of chunk too large might slow down the answer time and increase costs')
if 'chunk_length' not in st.session_state:
    st.session_state['chunk_length'] = st.session_state['config_server']['chunk_length']
if 'numeric' not in st.session_state:
    st.session_state['numeric'] = st.session_state['config_server']['chunk_length']
    config_new_rag['chunk_length'] = st.session_state['numeric']
if 'indexing' not in st.session_state:
    st.session_state['indexing'] = False


def update_slider_from_num():
    st.session_state['chunk_length'] = st.session_state['numeric']
    st.session_state['indexing'] = True


def update_num_from_slider():
    st.session_state['numeric'] = st.session_state['chunk_length']
    st.session_state['indexing'] = True


st.slider(label='**Choose length of chunks for indexing phases:**', min_value=0, max_value=2000, step=10, value=st.session_state['chunk_length'], help=' Keep in mind that too long or too short chunks can make the retrieval harder and decrease accuracy ', key='chunk_length', on_change=update_num_from_slider)
if 'all_system_prompt' not in st.session_state:
    st.session_state['all_system_prompt'] = st.session_state['config_server']['all_system_prompt']
system_prompt = st.selectbox(label='Choose the system prompt', options=st.session_state['all_system_prompt'].keys())
config_new_rag['name'] = st.text_input(label='**Give your custom RAG a name:**', placeholder='Enter name')
if st.button('**Create Custom RAG**', type='primary', help='Your RAG will only be visible in the LLM host at the moment of its creation, if you change LLM host, you must create your RAG again', use_container_width=True):
    all_rags_name = st.session_state['all_rags']
    if config_new_rag['name'] in all_rags_name:
        st.error(f"{config_new_rag['name']} already exists, please choose another name", icon='🚨')
    elif not bool(re.fullmatch('^[a-z0-9_-]+$', config_new_rag['name'])):
        st.error('Invalid RAG name, only alphanumeric characters, underscores, lowercase letters and hyphens are allowed', icon='🚨')
    else:
        st.session_state['custom_rags'].append(config_new_rag['name'])
        local_params = {'generation_system_prompt_name': system_prompt, 'forced_system_prompt': False}
        config_new_rag['local_params'] = local_params
        config_new_rag['chunk_length'] = st.session_state['chunk_length']
        
        try:
            _client.create_custom_rag(config_new_rag['name'], config_new_rag)
            st.session_state['all_rags'][config_new_rag['name']] = config_new_rag['name']
            _client.update_all_rags(st.session_state['all_rags'])
            st.session_state['benchmark']['rags'][config_new_rag['name']] = False
            st.session_state['rags_to_merge']['rags'][config_new_rag['name']] = False
            st.success('RAG successfully created')
        except APIError as e:
            st.error(f"Error creating custom RAG: {e}")

st.markdown('# Manage Custom RAGs:')
left, right = st.columns([0.85, 0.15], vertical_alignment='bottom')
rag_to_del = left.selectbox(label='List of custom RAGs', options=st.session_state['custom_rags'], label_visibility='collapsed')
if right.button(label='Delete RAG', type='primary', use_container_width=True):
    st.session_state['custom_rags'].remove(rag_to_del)
    if rag_to_del in st.session_state['all_rags'].keys():
        del st.session_state['all_rags'][rag_to_del]
    try:
        _client.delete_custom_rag(rag_to_del)
        _client.update_all_rags(st.session_state['all_rags'])
        _client.delete_elasticsearch_indices_by_prefix(rag_to_del)
        _client.delete_storage_by_prefix(rag_to_del)
    except APIError as e:
        st.error(f"Error deleting custom RAG: {e}")

if rag_to_del in st.session_state['custom_rags']:
    try:
        config = _client.get_custom_rag(rag_to_del)
        retrieval_methods = {'embeddings': 'Embeddings', 'bm25': 'BM25', 'hybrid': 'Hybrid'}
        display_config = {'Base RAG': [st.session_state['all_rags'][config['base']]], 'Vectorbase Type': [backend_vectorbase[config['params_vectorbase']['backend']]], 'Retrieval Method': [retrieval_methods[config['type_retrieval']]], 'Splitter': [splitter_dic[config['TextSplitter']]], 'Embedding model': [config['embedding_model']], 'Nb chunks': [str(config['nb_chunks'])], 'Chunk length': [str(config['chunk_length'])]}
        st.write(pd.DataFrame(display_config))
    except APIError as e:
        st.error(f"Error loading custom RAG config: {e}")
    
    st.markdown('# Manage Indexations:')
if 'rerun_managed_rag' not in st.session_state:
    st.session_state['rerun_managed_rag'] = False


def rerun_managed_rag():
    st.session_state['rerun_managed_rag'] = True
    st.session_state['indexation'] = None


st.selectbox(label='**Choose a RAG to view linked indexations**', options=base_rags.keys(), format_func=lambda x: base_rags[x], key='managed_rag', on_change=rerun_managed_rag)
rag_method = st.session_state['managed_rag']

try:
    custom_config = _client.get_custom_rag(rag_method)
    result = _client.generate_rag_names(rag_name=custom_config.get('base', rag_method), config=custom_config, additional_name='')
    names = result.get('names', [])
    base = custom_config.get('base', rag_method)
except APIError:
    try:
        result = _client.generate_rag_names(rag_method, st.session_state['config_server'], '')
        names = result.get('names', [])
    except APIError as e:
        st.error(f"API error: {e}")
        names = []
    cleaned_names = []
    for n in names:
        pointer = n.find('_rag')
        cleaned_names.append(n[:pointer] if pointer != -1 else n)
    names = cleaned_names
    base = rag_method

list_indexation = []
try:
    all_indices_response = _client.list_elasticsearch_indices()
    all_indices = all_indices_response.get('indices', [])
    for index_name in all_indices:
        if any((index_name.startswith(prefix) for prefix in names)):
            list_indexation.append(index_name)
except APIError as e:
    st.warning(f"Could not fetch elasticsearch indices: {e}")

left, right = st.columns([0.85, 0.15])
st.session_state['indexation'] = left.selectbox(label='indexations', label_visibility='collapsed', options=list_indexation)
if right.button(label='Delete indexation', use_container_width=True, type='primary'):
    if st.session_state.get('indexation'):
        try:
            if base == 'graph':
                indexation = st.session_state.indexation
                if '_local' in indexation:
                    pointer = indexation.find('_local')
                    suffix = '_global_search'
                elif '_global' in indexation:
                    pointer = indexation.find('_global')
                    suffix = '_local_search'
                else:
                    pointer = -1
                    suffix = ''
                
                _client.delete_elasticsearch_index(indexation)
                if pointer != -1 and suffix:
                    _client.delete_elasticsearch_index(indexation[:pointer] + suffix)
                
                if pointer != -1:
                    _client.delete_storage_by_prefix(indexation[:pointer])
                else:
                    _client.delete_storage_by_prefix(indexation)
            else:
                _client.delete_elasticsearch_index(st.session_state.indexation)
                _client.delete_storage_by_prefix(st.session_state.indexation)
            
            st.session_state['rerun_managed_run'] = False
            st.session_state.indexation = None
            st.rerun()
        except APIError as e:
            st.error(f"Error deleting indexation: {e}")

st.markdown('# Combine Responses from Different RAGs')
st.markdown('## Pick the RAGs You Want to Merge:')
config_merge_rag = {}
rags_to_merge_list = []
rags_config_to_merge_list = []
col1, col2, col3 = st.columns(3)
all_rags_list = list(st.session_state['all_rags'].keys())
nb_rags = len(all_rags_list)
rags_per_column = nb_rags // 3 if nb_rags % 3 == 0 else nb_rags // 3 + 1
with col1:
    for i in range(rags_per_column):
        if all_rags_list[i] == 'main' and st.session_state.hf_token in [None, '']:
            disable = True
        else:
            disable = False
        st.session_state['rags_to_merge']['rags'][all_rags_list[i]] = st.checkbox(label=st.session_state['all_rags'][all_rags_list[i]], value=st.session_state['rags_to_merge']['rags'][all_rags_list[i]], disabled=disable)
        if st.session_state['rags_to_merge']['rags'][all_rags_list[i]]:
            rags_to_merge_list.append(all_rags_list[i])
            rags_config_to_merge_list.append(get_config_rag(rag_name=all_rags_list[i]))
with col2:
    for i in range(rags_per_column, 2 * rags_per_column):
        if all_rags_list[i] == 'main' and st.session_state.hf_token in [None, '']:
            disable = True
        else:
            disable = False
        st.session_state['rags_to_merge']['rags'][all_rags_list[i]] = st.checkbox(label=st.session_state['all_rags'][all_rags_list[i]], value=st.session_state['rags_to_merge']['rags'][all_rags_list[i]], disabled=disable)
        if st.session_state['rags_to_merge']['rags'][all_rags_list[i]]:
            rags_to_merge_list.append(all_rags_list[i])
            rags_config_to_merge_list.append(get_config_rag(rag_name=all_rags_list[i]))
with col3:
    for i in range(2 * rags_per_column, nb_rags):
        if all_rags_list[i] == 'main' and st.session_state.hf_token in [None, '']:
            disable = True
        else:
            disable = False
        st.session_state['rags_to_merge']['rags'][all_rags_list[i]] = st.checkbox(label=st.session_state['all_rags'][all_rags_list[i]], value=st.session_state['rags_to_merge']['rags'][all_rags_list[i]], disabled=disable)
        if st.session_state['rags_to_merge']['rags'][all_rags_list[i]]:
            rags_to_merge_list.append(all_rags_list[i])
            rags_config_to_merge_list.append(get_config_rag(rag_name=all_rags_list[i]))
config_merge_rag['name'] = st.text_input(label='**Give your merge a name:**', placeholder='Enter name')
if st.button('**Create merge**', type='primary', help='Your merge will only be visible in the LLM host at the moment of its creation, if you change LLM host, you must create your merge again', use_container_width=True):
    if config_merge_rag['name'] in st.session_state['all_rags']:
        st.error(f"{config_merge_rag['name']} already exists, please choose another name", icon='🚨')
    elif not bool(re.fullmatch('^[a-z0-9_-]+$', config_merge_rag['name'])):
        st.error('Invalid RAG name, only alphanumeric characters, underscores, lowercase letters and hyphens are allowed', icon='🚨')
    else:
        st.session_state['merge_rags'].append(config_merge_rag['name'])
        saved_config = st.session_state['config_server'].copy()
        saved_config['name'] = config_merge_rag['name']
        saved_config['base'] = 'merger'
        saved_config['rag_list'] = rags_to_merge_list
        saved_config['rag_config_list'] = rags_config_to_merge_list
        
        try:
            _client.create_merge_rag(config_merge_rag['name'], rags_to_merge_list, rags_config_to_merge_list, saved_config)
            st.session_state['all_rags'][config_merge_rag['name']] = config_merge_rag['name']
            _client.update_all_rags(st.session_state['all_rags'])
            st.session_state['rags_to_merge']['rags'][config_merge_rag['name']] = False
            st.session_state['benchmark']['rags'][config_merge_rag['name']] = False
            st.success('RAG successfully created')
        except APIError as e:
            st.error(f"Error creating merge RAG: {e}")

st.markdown('## Manage Merged RAGs')
left, right = st.columns([0.85, 0.15], vertical_alignment='bottom')
rag_to_del = left.selectbox(label='List of merged RAGs', options=st.session_state.get('merge_rags', []), label_visibility='collapsed', key='rag_to_delete_selectbox')
if right.button(label='Delete merge', type='primary', use_container_width=True):
    if rag_to_del in st.session_state['merge_rags']:
        st.session_state['merge_rags'].remove(rag_to_del)
    if rag_to_del in st.session_state['all_rags'].keys():
        del st.session_state['all_rags'][rag_to_del]
    try:
        _client.delete_merge_rag(rag_to_del)
        _client.update_all_rags(st.session_state['all_rags'])
        st.success(f"✅ '{rag_to_del}' has been deleted.")
    except APIError as e:
        st.error(f"Error deleting merge RAG: {e}")
