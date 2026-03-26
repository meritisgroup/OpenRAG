import streamlit as st
from streamlit_.api_client import APIClient
from streamlit_.api_client.exceptions import APIError
from streamlit_.core.config import API_BASE_URL

_client = APIClient(API_BASE_URL)

st.markdown('# Prompt Management')

if 'editing_prompt' not in st.session_state:
    st.session_state['editing_prompt'] = None
if 'adding_prompt' not in st.session_state:
    st.session_state['adding_prompt'] = False
if 'prompt_to_delete' not in st.session_state:
    st.session_state['prompt_to_delete'] = None


def save_prompts():
    try:
        _client.update_config(st.session_state['config_server'])
        return True
    except APIError as e:
        st.error(f"Error saving configuration: {e}")
        return False


st.markdown('## Existing Prompts')

prompts = st.session_state['config_server']['all_system_prompt']
default_key = 'default'

for prompt_name, prompt_content in prompts.items():
    is_default = prompt_name == default_key
    is_editing = st.session_state['editing_prompt'] == prompt_name
    
    with st.container(border=True):
        col_header, col_actions = st.columns([4, 1])
        
        with col_header:
            st.markdown(f"**{prompt_name}**")
            if is_default:
                st.caption("_System default (cannot be deleted)_")
        
        with col_actions:
            if st.button("Edit", key=f"edit_btn_{prompt_name}", use_container_width=True):
                st.session_state['editing_prompt'] = prompt_name
                st.session_state['adding_prompt'] = False
                st.rerun()
        
        if is_editing:
            new_content = st.text_area(
                label="Prompt content",
                value=prompt_content,
                key=f"edit_content_{prompt_name}",
                height=200,
                label_visibility="collapsed"
            )
            
            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button("Save", key=f"save_btn_{prompt_name}", type="primary", use_container_width=True):
                    st.session_state['config_server']['all_system_prompt'][prompt_name] = new_content
                    if save_prompts():
                        st.success('Prompt saved!')
                    st.session_state['editing_prompt'] = None
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key=f"cancel_btn_{prompt_name}", use_container_width=True):
                    st.session_state['editing_prompt'] = None
                    st.rerun()
        else:
            st.text(prompt_content)
        
        if not is_default and not is_editing:
            if st.button("Delete", key=f"del_btn_{prompt_name}", type="secondary", use_container_width=True):
                st.session_state['prompt_to_delete'] = prompt_name
                st.rerun()

if st.session_state['prompt_to_delete']:
    prompt_to_delete = st.session_state['prompt_to_delete']
    st.warning(f"Are you sure you want to delete **{prompt_to_delete}**?")
    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        if st.button("Yes, delete", key="confirm_delete", type="primary"):
            st.session_state['config_server']['all_system_prompt'].pop(prompt_to_delete, None)
            
            if st.session_state['config_server']['local_params']['generation_system_prompt_name'] == prompt_to_delete:
                st.session_state['config_server']['local_params']['generation_system_prompt_name'] = 'default'
            
            if save_prompts():
                st.success(f'Prompt "{prompt_to_delete}" deleted')
            st.session_state['prompt_to_delete'] = None
            st.rerun()
    with col_cancel:
        if st.button("Cancel", key="cancel_delete"):
            st.session_state['prompt_to_delete'] = None
            st.rerun()

st.markdown('---')

if st.session_state['adding_prompt']:
    with st.container(border=True):
        st.markdown('**Add New Prompt**')
        
        new_name = st.text_input(
            label="Prompt name",
            placeholder="Enter a unique name (alphanumeric, underscores, hyphens)",
            key="new_prompt_name"
        )
        
        new_content = st.text_area(
            label="Prompt content",
            placeholder="Enter your system prompt...",
            key="new_prompt_content",
            height=200
        )
        
        col_add, col_cancel = st.columns(2)
        with col_add:
            if st.button("Add Prompt", key="confirm_add", type="primary", use_container_width=True):
                if not new_name:
                    st.error("Please enter a name for the prompt.")
                elif not new_content:
                    st.error("Please enter the prompt content.")
                elif new_name in st.session_state['config_server']['all_system_prompt']:
                    st.error("A prompt with this name already exists.")
                elif '"' in new_name:
                    st.error("Prompt name cannot contain quotes.")
                else:
                    st.session_state['config_server']['all_system_prompt'][new_name] = new_content
                    if save_prompts():
                        st.success(f'Prompt "{new_name}" added!')
                    st.session_state['adding_prompt'] = False
                    st.rerun()
        
        with col_cancel:
            if st.button("Cancel", key="cancel_add", use_container_width=True):
                st.session_state['adding_prompt'] = False
                st.rerun()

elif st.session_state['prompt_to_delete'] is None:
    if st.button("Add New Prompt", type="primary", use_container_width=True):
        st.session_state['adding_prompt'] = True
        st.session_state['editing_prompt'] = None
        st.rerun()
