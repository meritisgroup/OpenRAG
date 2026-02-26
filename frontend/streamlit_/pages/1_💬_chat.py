import streamlit as st
import time
import numpy as np
from streamlit_.services import RAGService, ConfigService
from streamlit_.utils.chat_funcs import get_chat_agent, handle_click, reset_success_button, change_default_prompt, prepare_show_context

st.markdown('# OpenRAG by Meritis')
if 'button_clicked' not in st.session_state:
    st.session_state['button_clicked'] = False
if 'success' not in st.session_state:
    st.session_state['success'] = False

with st.sidebar:
    rag_list = st.session_state['all_rags'].copy()
    rag_list = rag_list.keys()
    if st.session_state['selected_rag_method'] not in rag_list:
        st.session_state['selected_rag_method'] = list(rag_list)[0]
    rag_method = st.selectbox('**What RAG Method do you want to try ?**', options=rag_list, format_func=lambda x: st.session_state['all_rags'][x], key='selected_rag_method', on_change=reset_success_button)
    if 'chat_database_name' not in st.session_state or st.session_state['chat_database_name'] is None:
        st.session_state['chat_database_name'] = []
    if len(st.session_state['all_databases']) == 0 or st.session_state['all_databases'] is None:
        st.warning('⚠️ No database available, please create one in the databases page')
    else:
        st.multiselect(label='**Choose Database(s) for retrieval**', options=st.session_state['all_databases'], key='chat_database_name')
    if 'all_system_prompt' not in st.session_state:
        st.session_state['all_system_prompt'] = st.session_state['config_server']['all_system_prompt']
    if 'system_prompt_selected' not in st.session_state:
        st.session_state['system_prompt_selected'] = 'default'

    def force_system_prompt():
        system_prompt_selected = st.session_state['system_prompt_selected']
        if system_prompt_selected != 'default':
            st.session_state['config_server']['local_params']['forced_system_prompt'] = True
            st.session_state['config_server']['local_params']['generation_system_prompt_name'] = system_prompt_selected
        else:
            st.session_state['config_server']['local_params']['forced_system_prompt'] = False
        try:
            ConfigService.update_local_params(
                forced_system_prompt=st.session_state['config_server']['local_params']['forced_system_prompt'],
                generation_system_prompt_name=st.session_state['config_server']['local_params']['generation_system_prompt_name']
            )
        except Exception as e:
            pass
    
    change_default_prompt()
    system_prompt_selected = st.selectbox(label='**Choose system prompt**', options=st.session_state['all_system_prompt'].keys(), key='system_prompt_selected', on_change=force_system_prompt)
    reset_index = st.checkbox(label='Reset indexing', value=False)
    reset_preprocess = st.checkbox(label='Reset preprocessing', value=False)
    
    if st.button('Initialize RAG Agent', use_container_width=True, on_click=handle_click, disabled=st.session_state['button_clicked'], type='primary'):
        with st.spinner('Setting up RAG agent', show_time=True):
            session_id = get_chat_agent(rag_method=rag_method, databases_name=st.session_state['chat_database_name'])
        with st.spinner('Indexation running', show_time=True):
            try:
                client = st.session_state.get('api_client')
                if client:
                    RAGService.set_client(client)
                RAGService.run_indexation(
                    session_id=session_id,
                    reset_index=reset_index,
                    reset_preprocess=reset_preprocess
                )
            except Exception as e:
                st.error(f"Indexation error: {e}")
            st.session_state['success'] = True
            st.session_state['rag_name'] = rag_method
        st.session_state['button_clicked'] = False
        st.rerun()
    
    if st.session_state['success']:
        st.success('Indexation terminée !')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

if (prompt := st.chat_input('Waiting for the RAG to be initialed' if not st.session_state['success'] else 'Write Query', disabled=not st.session_state['success'])):
    st.session_state.messages.append({'role': 'user', 'content': '**User:** \n ' + prompt})
    with st.chat_message('user'):
        st.write('**User:**  \n' + prompt)
    with st.chat_message('assistant'):
        empty = st.empty()
        formated_answer = f"**{st.session_state['all_rags'][st.session_state['rag_name']]}:**  \n"
        empty.write(formated_answer)
        with st.spinner(text='*Computing answer*', show_time=True):
            start_time = time.time()
            try:
                client = st.session_state.get('api_client')
                if client:
                    RAGService.set_client(client)
                session_id = st.session_state.get('api_session_id')
                answer = RAGService.generate_answer(
                    query=prompt,
                    nb_chunks=5,
                    session_id=session_id if session_id else None
                )
            except Exception as e:
                answer = {
                    'answer': f"Error: {e}",
                    'nb_input_tokens': 0,
                    'nb_output_tokens': 0,
                    'context': '',
                    'impacts': [0, 0, ''],
                    'energy': [0, 0, '']
                }
            impacts = f"Between {np.round(answer['impacts'][0] * 1000, 2)} and {np.round(answer['impacts'][1] * 1000, 2)} {answer['impacts'][2][1:]}" if answer['impacts'] != [0, 0, ''] else 'Only measurable with Mistral and OpenAI LLM host'
            energy = f"Between {np.round(answer['energy'][0] * 1000, 2)} and {np.round(answer['energy'][1] * 1000, 2)} {answer['energy'][2][1:]}" if answer['energy'] != [0, 0, ''] else 'Only measurable with Mistral and OpenAI LLM host'
            end_time = time.time()
        if type(answer['context']) == list or type(answer['context']) == np.ndarray:
            context = prepare_show_context(chunks=answer['context'])
        else:
            context = answer['context']
        st.markdown("\n            <style>\n            /* Style de l'expander */\n            .streamlit-expanderHeader {\n                background-color:\n                color: white !important;\n                font-weight: normal !important;\n                border-radius: 8px;\n                padding: 8px;\n                text-transform: lowercase !important;\n            }\n            .streamlit-expanderHeader:hover {\n                background-color:\n                  /* vert un peu plus foncé au hover */\n            }\n            </style>\n            ", unsafe_allow_html=True)
        with st.expander('💡 Voir le contexte utilisé'):
            st.text(str(context))
        formated_answer += f"{answer['answer']} \n\n**Input tokens:** {answer['nb_input_tokens']}  \n**Output tokens:** {answer['nb_output_tokens']}  \n**Computing time:** {end_time - start_time:.2f} seconds  \n**Greenhouse gas emissions:**  {impacts}  \n**Power consumption:** {energy}"
        empty.write(formated_answer)
        st.session_state.messages.append({'role': 'ai', 'content': formated_answer})
