import streamlit as st
import time
import numpy as np
from backend.factory_RagAgent import (
    change_local_parameters,
    put_default_local_parameters,
    change_local_parameters,
)
from streamlit_.utils.chat_funcs import (
    get_chat_agent,
    handle_click,
    reset_success_button,
    change_default_prompt,
    prepare_show_context
)

st.markdown("# OpenRAG by Meritis")
if "button_clicked" not in st.session_state:
    st.session_state["button_clicked"] = False

if "success" not in st.session_state:
    st.session_state["success"] = False

with st.sidebar:
    rag_list = st.session_state["all_rags"][
        st.session_state["config_server"]["params_host_llm"]["type"]
    ].copy()

    rag_list = rag_list.keys()

    if st.session_state["rag_name"] not in rag_list:
        st.session_state["rag_name"] = list(rag_list)[0]
    rag_method = st.selectbox(
        "**What RAG Method do you want to try ?**",
        options=rag_list,
        format_func=lambda x: st.session_state["all_rags"][
            st.session_state["config_server"]["params_host_llm"]["type"]
        ][x],
        index=list(
            st.session_state["all_rags"][
                st.session_state["config_server"]["params_host_llm"]["type"]
            ].keys()
        ).index(st.session_state["rag_name"]),
        on_change=reset_success_button,
    )

    if "chat_database_name" not in st.session_state:
        st.session_state["chat_database_name"] = []

    st.multiselect(
        label="**Choose Database(s) for retrieval**",
        options=st.session_state["all_databases"],
        key="chat_database_name",
    )

    if "all_system_prompt" not in st.session_state:
        st.session_state["all_system_prompt"] = st.session_state["config_server"][
            "all_system_prompt"
        ]
    if "system_prompt_selected" not in st.session_state:
        st.session_state["system_prompt_selected"] = "default"

    def force_system_prompt():
        system_prompt_selected = st.session_state["system_prompt_selected"]
        if system_prompt_selected != "default":
            st.session_state["config_server"]["local_params"][
                "forced_system_prompt"
            ] = True
            st.session_state["config_server"]["local_params"][
                "generation_system_prompt_name"
            ] = system_prompt_selected
        else:
            st.session_state["config_server"]["local_params"][
                "forced_system_prompt"
            ] = False
        change_local_parameters(st.session_state["config_server"]["local_params"])

    change_default_prompt()
    system_prompt_selected = st.selectbox(
        label="**Choose system prompt**",
        options=st.session_state["all_system_prompt"].keys(),
        key="system_prompt_selected",
        on_change=force_system_prompt,
    )

    reset_index = st.checkbox(label="Reset indexing", value=False)

    if st.button(
        "Initialize RAG Agent",
        use_container_width=True,
        on_click=handle_click,
        disabled=st.session_state["button_clicked"],
        type="primary",
    ):
        with st.spinner("Setting up RAG agent", show_time=True):
            rag_agent = get_chat_agent(rag_method=rag_method,
                                       databases_name=st.session_state["chat_database_name"])

        with st.spinner("Indexation running", show_time=True):
            rag_agent.indexation_phase(
                reset_index=reset_index,
            )
            st.session_state["success"] = True
            st.session_state["rag"] = rag_agent
            st.session_state["rag_name"] = rag_method

        st.session_state["button_clicked"] = False
        st.rerun()
    if st.session_state["success"]:
        st.success("Indexation terminée !")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input(
    (
        "Waiting for the RAG to be initialed"
        if not st.session_state["success"]
        else "Write Query"
    ),
    disabled=not st.session_state["success"],
):
    st.session_state.messages.append(
        {"role": "user", "content": "**User:** \n " + prompt}
    )
    with st.chat_message("user"):
        st.write("**User:**  \n" + prompt)

    with st.chat_message("assistant"):
        empty = st.empty()
        formated_answer = f"**{st.session_state['all_rags'][st.session_state['config_server']['params_host_llm']['type']][st.session_state['rag_name']]}:**  \n"
        empty.write(formated_answer)
        with st.spinner(text="*Computing answer*", show_time=True):
            start_time = time.time()
            answer = st.session_state["rag"].generate_answer(
                query=prompt, nb_chunks=st.session_state["rag"].nb_chunks
            )
            impacts = (
                f"Between {np.round(answer['impacts'][0]*1000,2)} and {np.round(answer['impacts'][1]*1000,2)} {answer['impacts'][2][1:]}"
                if answer["impacts"] != [0, 0, ""]
                else "Only measurable with Mistral and OpenAI LLM host"
            )
            energy = (
                f"Between {np.round(answer['energy'][0]*1000,2)} and {np.round(answer['energy'][1]*1000,2)} {answer['energy'][2][1:]}"
                if answer["energy"] != [0, 0, ""]
                else "Only measurable with Mistral and OpenAI LLM host"
            )
            end_time = time.time()

        context = prepare_show_context(context=answer["context"],
                                       docs_name=answer["docs_name"])

        st.markdown(
            """
            <style>
            /* Style de l'expander */
            .streamlit-expanderHeader {
                background-color: #4CAF50;  /* vert */
                color: white !important;
                font-weight: bold;
                border-radius: 8px;
                padding: 8px;
            }
            .streamlit-expanderHeader:hover {
                background-color: #45a049; /* vert un peu plus foncé au hover */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        with st.expander("💡 Voir le contexte utilisé"):
            st.write(context)

        formated_answer += (
            f"{answer['answer']} \n\n"
            f"**Input tokens:** {answer['nb_input_tokens']}  \n"
            f"**Output tokens:** {answer['nb_output_tokens']}  \n"
            f"**Computing time:** {(end_time - start_time):.2f} seconds  \n"
            f"**Greenhouse gas emissions:**  {impacts}  \n"
            f"**Power consumption:** {energy}"
        )
        empty.write(formated_answer)
        st.session_state.messages.append({"role": "ai", "content": formated_answer})
