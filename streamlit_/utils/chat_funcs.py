import streamlit as st
from backend.factory_RagAgent import get_rag_agent, get_custom_rag_agent, change_config_server
import json



def get_chat_agent(rag_method):
    if ("custom_rags" in st.session_state.keys() and rag_method in st.session_state["custom_rags"]):
        if st.session_state["config_server"]["params_host_llm"]["type"] == "vllm":
            folder = "vllm"
        else:
            folder = "ollama_openai_mistral"

        with open(f"data/custom_rags/{folder}/{rag_method}.json","r") as file:
            custom_config = json.load(file)
        custom_config["params_host_llm"] = st.session_state["config_server"]["params_host_llm"]
        custom_config = change_config_server(rag_name=rag_method, 
                                             config_server= custom_config)
        rag_agent = get_custom_rag_agent(custom_config["base"],
                                         config_server=custom_config,
                                         database_name=st.session_state["chat_database_name"])
    else:
        st.session_state["config_server"] = change_config_server(rag_name=rag_method, config_server=st.session_state["config_server"])
        rag_agent = get_rag_agent(rag_method,
                                   config_server=st.session_state["config_server"], database_name=st.session_state["chat_database_name"])

    return rag_agent

def handle_click():
    "Handles if the button Initialize Rag is clickable or not"
    st.session_state["button_clicked"] = True

def reset_success_button():
    "Deletes success button if a second indexation is started"
    st.session_state["success"] = False