import streamlit as st
import pandas as pd
import os
from backend.factory_RagAgent import (
    get_rag_agent,
    get_custom_rag_agent,
    change_config_server,
)
import json
from streamlit_.utils.benchmark_funcs import (
    generate_benchmark,
    display_plot,
    match_name_cleaner,
    clean_bench_df,
)
from streamlit_.utils.chat_funcs import get_chat_agent

from backend.utils.progress import ProgressBar

st.markdown("# Benchmark Generation")
st.markdown("## Choose RAG techniques to benchmark:")
col1, col2, col3 = st.columns(3)


all_rags = list(
    st.session_state["all_rags"][
        st.session_state["config_server"]["params_host_llm"]["type"]
    ].keys()
)
nb_rags = len(all_rags)
rags_per_column = nb_rags // 3 if nb_rags % 3 == 0 else nb_rags // 3 + 1

with col1:
    for i in range(rags_per_column):
        if all_rags[i] == "main" and st.session_state.hf_token in [None, ""]:
            disable = True
        else:
            disable = False
        st.session_state["benchmark"]["rags"][all_rags[i]] = st.checkbox(
            label=st.session_state["all_rags"][
                st.session_state["config_server"]["params_host_llm"]["type"]
            ][all_rags[i]],
            value=st.session_state["benchmark"]["rags"][all_rags[i]],
            disabled=disable,
        )

with col2:
    for i in range(rags_per_column, 2 * rags_per_column):
        if all_rags[i] == "main" and st.session_state.hf_token in [None, ""]:
            disable = True
        else:
            disable = False
        st.session_state["benchmark"]["rags"][all_rags[i]] = st.checkbox(
            label=st.session_state["all_rags"][
                st.session_state["config_server"]["params_host_llm"]["type"]
            ][all_rags[i]],
            value=st.session_state["benchmark"]["rags"][all_rags[i]],
            disabled=disable,
        )

with col3:
    for i in range(2 * rags_per_column, nb_rags):
        if all_rags[i] == "main" and st.session_state.hf_token in [None, ""]:
            disable = True
        else:
            disable = False
        st.session_state["benchmark"]["rags"][all_rags[i]] = st.checkbox(
            label=st.session_state["all_rags"][
                st.session_state["config_server"]["params_host_llm"]["type"]
            ][all_rags[i]],
            value=st.session_state["benchmark"]["rags"][all_rags[i]],
            disabled=disable,
        )

st.markdown("## Import your list of queries")
left, right = st.columns([0.5, 0.15], vertical_alignment="bottom")
list_queries = os.listdir("./data/queries/")
if ".gitkeep" in list_queries:
    list_queries.remove(".gitkeep")
st.session_state["benchmark"]["queries_doc_name"] = left.selectbox(
    label="Select your list of queries",
    options=list_queries,
    label_visibility="collapsed",
)
if right.button(label="Delete Query Doc", type="primary", use_container_width=True):
    os.remove("./data/queries/" + st.session_state["benchmark"]["queries_doc_name"])
    st.rerun()

if st.session_state["benchmark"]["queries_doc_name"] is not None:
    st.session_state["benchmark"]["queries"] = pd.read_excel(
        "./data/queries/" + st.session_state["benchmark"]["queries_doc_name"],
        index_col=0,
        dtype={"query": str, "answer": str},
        names=["query", "answer"],
    )
    st.write(st.session_state["benchmark"]["queries"])


def set_bool():
    st.session_state["benchmark"]["load"] = True


uploaded_files = st.file_uploader(
    "Only `Excel` files are supported",
    type=["xls", "xlsx", "xlsm", "odt", "xlsb"],
    accept_multiple_files=False,
    on_change=set_bool,
)

if st.session_state["benchmark"]["load"]:
    os.makedirs("./data/queries", exist_ok=True)
    path = "./data/queries/" + uploaded_files.name
    queries = pd.read_excel(
        uploaded_files,
        engine="openpyxl",
        index_col=0,
        dtype={"query": str, "answer": str},
        names=["query", "answer"],
    )
    queries.to_excel(path)
    st.session_state["benchmark"]["load"] = False
    st.rerun()


st.markdown("## Choose database to perfom benchmark on")
st.selectbox(
    "Choose database",
    options=st.session_state["all_databases"],
    label_visibility="collapsed",
    index=(
        st.session_state["all_databases"].index(st.session_state["benchmark_database"])
        if st.session_state["benchmark_database"] is not None
        else None
    ),
    key="benchmark_database",
)

reset_index = st.checkbox(label="Reset indexing", value=False)

if "benchmark_clicked" not in st.session_state:
    st.session_state["benchmark_clicked"] = False
    st.session_state["plot_to_display"] = False


def handle_click():
    st.session_state["benchmark_clicked"] = True
    st.set_option("client.showSidebarNavigation", False)
    st.sidebar.page_link(
        "streamlit_/pages/1_💬_chat.py",
        disabled=True,
        help="❌ Unaccessible during benchmark. Refresh page to interrupt process",
        icon="💬",
    )
    st.sidebar.page_link(
        "streamlit_/pages/2_🧠_configuration.py",
        disabled=True,
        help="❌ Unaccessible during benchmark. Refresh page to interrupt process",
        icon="🧠",
    )
    st.sidebar.page_link(
        "streamlit_/pages/3_📚_benchmark.py",
        disabled=True,
        help="❌ Unaccessible during benchmark. Refresh page to interrupt process",
        icon="📚",
    )
    st.sidebar.page_link(
        "streamlit_/pages/4_🔧_rag_maker.py",
        disabled=True,
        help="❌ Unaccessible during benchmark. Refresh page to interrupt process",
        icon="🔧",
    )
    st.sidebar.page_link(
        "streamlit_/pages/5_🌐_databases.py",
        disabled=True,
        help="❌ Unaccessible during benchmark. Refresh page to interrupt process",
        icon="🌐",
    )
    st.sidebar.page_link(
        "streamlit_/pages/6_📖_documentation.py",
        disabled=True,
        help="❌ Unaccessible during benchmark. Refresh page to interrupt process",
        icon="📖",
    )


col1, col2, col3, col4 = st.columns([0.20, 0.20, 0.20, 0.40])
if col1.button(
    "Generate Benchmark",
    on_click=handle_click,
    disabled=st.session_state["benchmark_clicked"],
    use_container_width=True,
    type="primary",
):
    if st.session_state["benchmark_database"] is None:
        st.session_state["benchmark_clicked"] = False
        st.error("Choose a database")
    else:

        rag_agents = []
        rag_names = []
        with st.spinner("**Indexation Running**", show_time=True):
            progress_bar_iterable = [
                rag
                for rag in st.session_state["benchmark"]["rags"].keys()
                if st.session_state["benchmark"]["rags"][rag]
            ]
            indexation_progress_bar = ProgressBar(progress_bar_iterable)
            for i, rag in enumerate(indexation_progress_bar.iterable):
                if st.session_state["benchmark"]["rags"][rag]:
                    rag_agent = get_chat_agent(rag)

                    rag_agent.indexation_phase(
                        f"./data/databases/{st.session_state['benchmark_database']}",
                        reset_index,
                    )
                    rag_agents.append(rag_agent)
                    rag_names.append(rag)
                    indexation_progress_bar.update(i)

        if len(rag_agents) > 1:
            with st.spinner(
                f"**Generating benchmark for the following RAGs:** {rag_names}",
                show_time=True,
            ):
                indexation_progress_bar.success("Indexation done")
                generate_benchmark(rag_names, rag_agents)
                st.session_state["benchmark_clicked"] = False
                st.set_option("client.showSidebarNavigation", True)
                st.rerun()
        else:
            st.session_state["benchmark_clicked"] = False
            st.error("Choose at least 2 rags methods")


if "report_path" in st.session_state["benchmark"]:
    st.success(
        "***Benchmark terminé !***",
    )
    with open(
        st.session_state["benchmark"]["report_path"] + "/plot_report.pdf", "rb"
    ) as file:
        col2.download_button(
            label="Download report",
            data=file,
            file_name="report.pdf",
            type="primary",
            use_container_width=True,
        )
    clean_bench_df()
    with open(
        st.session_state["benchmark"]["report_path"] + "/answers.xlsx", "rb"
    ) as file:
        col3.download_button(
            label="Download answers",
            data=file,
            file_name="benchmark_answers.xlsx",
            type="primary",
            use_container_width=True,
        )

    display_plot()
    with open(
        st.session_state["benchmark"]["report_path"] + "/config_server.json", "w"
    ) as file:
        json.dump(st.session_state["config_server"], file, indent=4)
    match = st.selectbox(
        label="**Choose arena match to analyse**",
        options=st.session_state["benchmark"]["matches"],
        format_func=match_name_cleaner,
    )
    st.plotly_chart(st.session_state["benchmark"]["plots"]["arena_graphs"][match])
