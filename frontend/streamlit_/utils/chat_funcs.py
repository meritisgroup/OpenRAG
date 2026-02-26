import streamlit as st
import re
from streamlit_.services import RAGService


def get_chat_agent(rag_method, databases_name, session_state=None):
    if session_state is None:
        session_state = st.session_state
    
    client = session_state.get('api_client')
    if client:
        RAGService.set_client(client)
    
    config = session_state['config_server'].copy()
    
    if 'custom_rags' in session_state.keys() and rag_method in session_state['custom_rags']:
        custom_config = client.get_custom_rag(rag_method)
        custom_config['params_host_llm'] = config['params_host_llm']
        base_rag = custom_config.get('base', rag_method)
        session_id = RAGService.get_chat_agent(
            rag_method=base_rag,
            databases_name=databases_name,
            config_server=custom_config,
            models_infos=session_state['models_infos']
        )
    elif 'merge_rags' in session_state.keys() and rag_method in session_state['merge_rags']:
        merge_config = client.get_merge_rag(rag_method)
        merge_config['params_host_llm'] = config['params_host_llm']
        session_id = RAGService.get_chat_agent(
            rag_method='merger',
            databases_name=databases_name,
            config_server=merge_config,
            models_infos=session_state['models_infos']
        )
    else:
        session_id = RAGService.get_chat_agent(
            rag_method=rag_method,
            databases_name=databases_name,
            config_server=config,
            models_infos=session_state['models_infos']
        )
    
    session_state['api_session_id'] = session_id
    return session_id


def change_default_prompt():
    try:
        default_value = st.session_state['config_server']['local_params']['generation_system_prompt_name']
        st.session_state['system_prompt_selected'] = default_value
    except Exception as e:
        st.session_state['system_prompt_selected'] = 'default'


def handle_click():
    st.session_state['button_clicked'] = True


def reset_success_button():
    st.session_state['success'] = False


def clean_markdown(text: str) -> str:
    return re.sub('\\*{1,2}', '', text)


def prepare_show_context(chunks):
    blocks = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            cleaned_context = clean_markdown(chunk.get('text', ''))
            block = ''
            if chunk.get('document'):
                block += f"source : {chunk['document']}\n\n"
            block += f'{cleaned_context}'
            if chunk.get('rerank_score') is not None:
                block += f"\n\n Rerank score : {chunk['rerank_score']}"
        else:
            cleaned_context = clean_markdown(chunk.text if hasattr(chunk, 'text') else str(chunk))
            block = ''
            if hasattr(chunk, 'document') and chunk.document is not None:
                block += f"source : {chunk.document}\n\n"
            block += f'{cleaned_context}'
            if hasattr(chunk, 'rerank_score') and chunk.rerank_score is not None:
                block += f"\n\n Rerank score : {chunk.rerank_score}"
        blocks.append(block)
    separator = '\n' + '-' * 200 + '\n'
    output = separator.join(blocks)
    return output
