import streamlit as st
from backend.factory_RagAgent import (
    get_rag_agent,
    get_custom_rag_agent,
    change_config_server,
)
import json


def get_chat_agent(rag_method, databases_name, session_state=None):
    if session_state is None:
        session_state = st.session_state

    if (
        "custom_rags" in session_state.keys()
        and rag_method in session_state["custom_rags"]
    ):
        folder = session_state["config_server"]["params_host_llm"]["type"]
        with open(f"data/custom_rags/{folder}/{rag_method}.json", "r") as file:
            custom_config = json.load(file)

        custom_config["params_host_llm"] = session_state["config_server"][
            "params_host_llm"
        ]

        rag_agent = get_custom_rag_agent(
            custom_config["base"],
            config_server=custom_config,
            databases_name=databases_name,
        )
    elif (
        "merge_rags" in session_state.keys()
        and rag_method in session_state["merge_rags"]
    ):
        folder = session_state["config_server"]["params_host_llm"]["type"]
        with open(f"data/merge/{folder}/{rag_method}.json", "r") as file:
            custom_config = json.load(file)

        custom_config["params_host_llm"] = session_state["config_server"][
            "params_host_llm"
        ]

        rag_agent = get_custom_rag_agent(
            "merger",
            config_server=custom_config,
            databases_name=databases_name,
        )
    else:
        session_state["config_server"] = change_config_server(
            rag_name=rag_method, config_server=session_state["config_server"]
        )
        rag_agent = get_rag_agent(
            rag_method,
            config_server=session_state["config_server"],
            databases_name=databases_name,
        )

    return rag_agent


def change_default_prompt():
    try:
        default_value = st.session_state["config_server"]["local_params"][
            "generation_system_prompt_name"
        ]
        st.session_state["system_prompt_selected"] = default_value
    except Exception as e:
        st.session_state["system_prompt_selected"] = "default"


def handle_click():
    "Handles if the button Initialize Rag is clickable or not"
    st.session_state["button_clicked"] = True


def reset_success_button():
    "Deletes success button if a second indexation is started"
    st.session_state["success"] = False



def prepare_show_context(context, docs_name):
    context = context.split("\n[...]\n")
    context = [c for c in context if c.strip() != ""]

    if len(docs_name)!=len(context):
        blocks = []
        for i in range(len(context)):
            cleaned_context = context[i].lstrip("\n")  
            block = f"{cleaned_context}"
            blocks.append(block)
            
        output = "\n\n---\n\n".join(blocks)
        return output
    else:
        blocks = []
        for i in range(len(docs_name)):
            cleaned_context = context[i].lstrip("\n")  
            block = f"source : {docs_name[i]}\n\n{cleaned_context}"
            blocks.append(block)
            
        output = "\n\n---\n\n".join(blocks)
        return output
