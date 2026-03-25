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
if 'multi_rag_sessions' not in st.session_state:
    st.session_state['multi_rag_sessions'] = {}
if 'selected_rag_methods' not in st.session_state:
    st.session_state['selected_rag_methods'] = []

with st.sidebar:
    rag_list = st.session_state['all_rags'].copy()
    rag_list = list(rag_list.keys())
    
    if not rag_list:
        st.warning('⚠️ Backend not connected or no RAG method available')
        st.stop()
    
    if 'rags_availability' not in st.session_state:
        with st.spinner("Checking models availability..."):
            try:
                client = st.session_state.get('api_client')
                if client:
                    RAGService.set_client(client)
                availability = RAGService.get_rags_availability(
                    config=st.session_state['config_server'],
                    models_infos=st.session_state['models_infos']
                )
                st.session_state['rags_availability'] = availability.get('rags', {})
            except Exception as e:
                st.session_state['rags_availability'] = {}
                st.warning(f"Could not check RAG availability: {e}")
    
    rags_availability = st.session_state.get('rags_availability', {})
    available_rags = {k: st.session_state['all_rags'][k] for k in rag_list 
                      if rags_availability.get(k, {}).get('available', True)}
    unavailable_rags = {k: st.session_state['all_rags'][k] for k in rag_list 
                        if not rags_availability.get(k, {}).get('available', True)}
    
    if not available_rags:
        st.error('⚠️ No RAG available with current model configuration')
        if unavailable_rags:
            with st.expander(f"📋 {len(unavailable_rags)} RAG(s) unavailable"):
                for rag_id, rag_name in unavailable_rags.items():
                    rag_info = rags_availability.get(rag_id, {})
                    missing = rag_info.get('missing_models', {})
                    if missing:
                        missing_str = ", ".join([f"{k}" for k in missing.keys()])
                        st.caption(f"**{rag_name}**: missing {missing_str}")
        if st.button("🔄 Refresh availability"):
            del st.session_state['rags_availability']
            st.rerun()
        st.stop()
    
    available_keys = list(available_rags.keys())
    
    default_rag = [available_keys[0]] if available_keys else []
    selected_rags = st.multiselect(
        '**Select RAG Method(s)**',
        options=available_keys,
        format_func=lambda x: available_rags[x],
        default=st.session_state.get('selected_rag_methods', default_rag),
        key='selected_rag_methods',
        on_change=reset_success_button
    )
    
    is_compare_mode = len(selected_rags) > 1
    if is_compare_mode:
        st.info(f"📊 Comparison mode: {len(selected_rags)} RAGs selected")
    
    if unavailable_rags:
        with st.expander(f"⚠️ {len(unavailable_rags)} RAG(s) unavailable"):
            for rag_id, rag_name in unavailable_rags.items():
                rag_info = rags_availability.get(rag_id, {})
                missing = rag_info.get('missing_models', {})
                if missing:
                    missing_str = ", ".join([f"{k}" for k in missing.keys()])
                    st.caption(f"**{rag_name}**: missing {missing_str}")
    
    if st.button("🔄 Refresh"):
        if 'rags_availability' in st.session_state:
            del st.session_state['rags_availability']
        st.rerun()
    
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
    
    button_label = f"Initialize {len(selected_rags)} RAG Agent(s)" if len(selected_rags) > 1 else "Initialize RAG Agent"
    if st.button(button_label, use_container_width=True, on_click=handle_click, disabled=st.session_state['button_clicked'] or len(selected_rags) == 0, type='primary'):
        if len(selected_rags) == 0:
            st.warning("Please select at least one RAG method")
            st.session_state['button_clicked'] = False
            st.stop()
        
        try:
            client = st.session_state.get('api_client')
            if client:
                RAGService.set_client(client)
            
            new_sessions = {}
            for rag_method in selected_rags:
                with st.spinner(f'Setting up {st.session_state["all_rags"][rag_method]}...'):
                    session_id = get_chat_agent(
                        rag_method=rag_method, 
                        databases_name=st.session_state['chat_database_name'],
                        create_new_session=True
                    )
                    new_sessions[rag_method] = session_id
            
            st.session_state['multi_rag_sessions'] = new_sessions
            
            for rag_method, session_id in new_sessions.items():
                RAGService.run_indexation(
                    session_id=session_id,
                    reset_index=reset_index,
                    reset_preprocess=reset_preprocess
                )
            
            progress_bars = {}
            status_placeholders = {}
            for rag_method in selected_rags:
                rag_name = st.session_state["all_rags"][rag_method]
                st.markdown(f"**{rag_name}**")
                progress_bars[rag_method] = st.progress(0, text=f"Starting {rag_name}...")
                status_placeholders[rag_method] = st.empty()
            
            all_completed = False
            while not all_completed:
                all_completed = True
                for rag_method, session_id in new_sessions.items():
                    status = RAGService.get_indexation_status(session_id=session_id)
                    progress_bar = progress_bars[rag_method]
                    status_placeholder = status_placeholders[rag_method]
                    rag_name = st.session_state["all_rags"][rag_method]
                    
                    if status['status'] == 'completed':
                        progress_bar.progress(100, text=f"{rag_name} completed!")
                        status_placeholder.empty()
                    elif status['status'] == 'error':
                        progress_bar.empty()
                        status_placeholder.error(f"{rag_name} failed: {status.get('error', 'Unknown error')}")
                        all_completed = False
                    elif status['status'] == 'running':
                        all_completed = False
                        progress = status.get('progress', 0)
                        message = status.get('message', 'Processing...')
                        progress_bar.progress(progress / 100, text=f"{rag_name}: {message}")
                        status_placeholder.info(message)
                    else:
                        all_completed = False
                
                if not all_completed:
                    time.sleep(3)
            
            st.session_state['success'] = True
            st.session_state['rag_name'] = selected_rags[0] if len(selected_rags) == 1 else None
        except Exception as e:
            st.error(f"Indexation error: {e}")
        st.session_state['button_clicked'] = False
        st.rerun()
    
    if st.session_state['success']:
        st.success('Indexation completed!')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state.messages:
    if message.get('is_comparison'):
        with st.chat_message(message['role']):
            st.markdown(f"**Question:** {message['prompt']}")
            cols = st.columns(len(message['results']))
            for i, (rag_method, result) in enumerate(message['results'].items()):
                with cols[i]:
                    rag_name = st.session_state['all_rags'].get(rag_method, rag_method)
                    answer = result['answer']
                    st.markdown(f"**{rag_name}**")
                    st.markdown("---")
                    st.markdown(answer['answer'])
                    st.caption(f"⏱️ {result['time']:.2f}s | 📊 {answer['nb_input_tokens']}/{answer['nb_output_tokens']} tokens")
                    
                    if type(answer['context']) == list or type(answer['context']) == np.ndarray:
                        context = prepare_show_context(chunks=answer['context'])
                    else:
                        context = answer['context']
                    with st.expander("💡 Context used"):
                        st.text(str(context)[:3000] + "..." if len(str(context)) > 3000 else str(context))
    else:
        with st.chat_message(message['role']):
            st.write(message['content'])

active_sessions = st.session_state.get('multi_rag_sessions', {})
is_compare_mode = len(active_sessions) > 1

if (prompt := st.chat_input('Waiting for the RAG to be initialed' if not st.session_state['success'] else 'Write Query', disabled=not st.session_state['success'])):
    st.session_state.messages.append({'role': 'user', 'content': '**User:** \n ' + prompt})
    with st.chat_message('user'):
        st.write('**User:**  \n' + prompt)
    
    client = st.session_state.get('api_client')
    if client:
        RAGService.set_client(client)
    
    if is_compare_mode:
        results = {}
        for rag_method, session_id in active_sessions.items():
            with st.spinner(f"Computing {st.session_state['all_rags'][rag_method]}..."):
                start_time = time.time()
                try:
                    answer = RAGService.generate_answer(
                        query=prompt,
                        nb_chunks=st.session_state['config_server'].get('nb_chunks', 5),
                        session_id=session_id
                    )
                except Exception as e:
                    answer = {
                        'answer': f"Error: {e}",
                        'nb_input_tokens': 0,
                        'nb_output_tokens': 0,
                        'context': '',
                        'impacts': [0, 0, ''],
                        'energy': [0, 0, ''],
                        'databases': []
                    }
                end_time = time.time()
                results[rag_method] = {
                    'answer': answer,
                    'time': end_time - start_time
                }
        
        st.markdown("---")
        st.markdown("### 📊 Comparison Results")
        
        cols = st.columns(len(results))
        
        for i, (rag_method, result) in enumerate(results.items()):
            answer = result['answer']
            rag_name = st.session_state['all_rags'][rag_method]
            
            with cols[i]:
                st.markdown(f"**{rag_name}**")
                st.markdown("---")
                st.markdown(answer['answer'])
                st.caption(f"⏱️ {result['time']:.2f}s | 📊 {answer['nb_input_tokens']}/{answer['nb_output_tokens']} tokens")
                
                if type(answer['context']) == list or type(answer['context']) == np.ndarray:
                    context = prepare_show_context(chunks=answer['context'])
                else:
                    context = answer['context']
                with st.expander("💡 Context used"):
                    st.text(str(context)[:3000] + "..." if len(str(context)) > 3000 else str(context))
        
        st.session_state.messages.append({
            'role': 'ai',
            'is_comparison': True,
            'prompt': prompt,
            'results': results
        })
    
    else:
        with st.chat_message('assistant'):
            empty = st.empty()
            rag_name = st.session_state['all_rags'].get(list(active_sessions.keys())[0], 'RAG') if active_sessions else 'RAG'
            formated_answer = f"**{rag_name}:**  \n"
            empty.write(formated_answer)
            with st.spinner(text='*Computing answer*'):
                start_time = time.time()
                try:
                    session_id = list(active_sessions.values())[0] if active_sessions else None
                    rag_config = st.session_state.get('active_rag_config', st.session_state['config_server'])
                    answer = RAGService.generate_answer(
                        query=prompt,
                        nb_chunks=rag_config.get('nb_chunks', 5),
                        session_id=session_id
                    )
                except Exception as e:
                    answer = {
                        'answer': f"Error: {e}",
                        'nb_input_tokens': 0,
                        'nb_output_tokens': 0,
                        'context': '',
                        'impacts': [0, 0, ''],
                        'energy': [0, 0, ''],
                        'databases': []
                    }
                db_names = answer.get('databases', [])
                db_str = f" ({', '.join(db_names)})" if db_names else ""
                impacts = f"Between {np.round(answer['impacts'][0] * 1000, 2)} and {np.round(answer['impacts'][1] * 1000, 2)} {answer['impacts'][2][1:]}" if answer['impacts'] != [0, 0, ''] else 'Only measurable with Mistral and OpenAI LLM host'
                energy = f"Between {np.round(answer['energy'][0] * 1000, 2)} and {np.round(answer['energy'][1] * 1000, 2)} {answer['energy'][2][1:]}" if answer['energy'] != [0, 0, ''] else 'Only measurable with Mistral and OpenAI LLM host'
                end_time = time.time()
            if type(answer['context']) == list or type(answer['context']) == np.ndarray:
                context = prepare_show_context(chunks=answer['context'])
            else:
                context = answer['context']
            with st.expander('💡 View context used'):
                st.text(str(context))
            formated_answer = f"**{rag_name}{db_str}:**  \n{answer['answer']} \n\n**Input tokens:** {answer['nb_input_tokens']}  \n**Output tokens:** {answer['nb_output_tokens']}  \n**Computing time:** {end_time - start_time:.2f} seconds  \n**Greenhouse gas emissions:**  {impacts}  \n**Power consumption:** {energy}"
            empty.write(formated_answer)
            st.session_state.messages.append({'role': 'ai', 'content': formated_answer})
