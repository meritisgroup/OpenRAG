import streamlit as st
import re
from backend.factory_RagAgent import (
    get_rag_agent,
    get_custom_rag_agent,
    change_config_server,
)
import json
from backend.database.rag_classes import Document, Tokens, Chunk


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


def clean_markdown(text: str) -> str:
    return re.sub(r"\*{1,2}", "", text)

def prepare_show_context(chunks: list[Chunk]):
    blocks = []
    for chunk in chunks:
        cleaned_context = clean_markdown(chunk.text)
        block = ""
        if chunk.document != None:
            block += f"source : {chunk.document}\n\n"
        block = f"{cleaned_context}"
        if chunk.rerank_score != None:
            block += f"\n\n Rerank score : {chunk.rerank_score}"
        blocks.append(block)

    output = "\n\n" + ("-" * 10) + "\n\n".join(blocks)
    return output
