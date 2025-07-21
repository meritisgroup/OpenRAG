import streamlit as st
from backend.factory_RagAgent import get_rag_agent, change_config_server
import json
import pandas as pd
import os
import glob
from dotenv import load_dotenv

load_dotenv()

chat = st.Page("streamlit_/pages/1_💬_chat.py", title="Chat")
config = st.Page("streamlit_/pages/2_🧠_configuration.py", title="Configuration")
benchmark = st.Page("streamlit_/pages/3_📚_benchmark.py", title="Benchmark")
rag_maker = st.Page("streamlit_/pages/4_🔧_rag_maker.py", title="Rag Maker")
databases = st.Page("streamlit_/pages/5_🌐_databases.py", title="Databases")
documentation = st.Page("streamlit_/pages/6_📖_documentation.py", title="Documentation")
advanced_configuration = st.Page(
    "streamlit_/pages/7_🛠️_advanced_configuration.py", title="Advanced Configuration"
)
pg = st.navigation(
    (
        [
            chat,
            config,
            benchmark,
            rag_maker,
            databases,
            documentation,
            advanced_configuration,
        ]
    )
)

st.set_page_config(
    page_title="OpenRAG by Meritis",
    page_icon="streamlit_/images/symbole_meritis.png",
    layout="wide",
)
st.set_option("client.showSidebarNavigation", True)

st.logo(
    "streamlit_/images/logomeritis_horizontal.png",
    size="large",
    link="https://meritis.fr/",
    icon_image="streamlit_/images/logomeritis_horizontal_rvb.png",
)

if "config_server" not in st.session_state:
    with open("streamlit_/utils/base_config_server.json", "r") as file:
        base_config_server = json.load(file)
        if base_config_server["params_vectorbase"]["backend"] == "elasticsearch":
            base_config_server["params_vectorbase"]["url"] = (
                os.getenv("ES_LOCAL_URL") + ":" + os.getenv("ES_LOCAL_PORT")
            )
            base_config_server["params_vectorbase"]["auth"] = [
                "elastic",
                os.getenv("ES_LOCAL_PASSWORD"),
            ]
        st.session_state["config_server"] = base_config_server
        st.session_state["hf_token"] = st.session_state["config_server"]["hf_token"]

    for file in glob.glob("data/custom_rags/ollama_openai_mistral/*.json"):
        with open(file, "r") as f:
            config = json.load(f)
        if config["params_vectorbase"]["backend"] == "elasticsearch":
            config["params_vectorbase"]["url"] = (
                os.getenv("ES_LOCAL_URL") + ":" + os.getenv("ES_LOCAL_PORT")
            )
            config["params_vectorbase"]["auth"] = [
                "elastic",
                os.getenv("ES_LOCAL_PASSWORD"),
            ]
        with open(file, "w") as f:
            json.dump(config, f, indent=4)

    for file in glob.glob("data/custom_rags/vllm/*.json"):
        with open(file, "r") as f:
            config = json.load(f)
        if config["params_vectorbase"]["backend"] == "elasticsearch":
            config["params_vectorbase"]["url"] = (
                os.getenv("ES_LOCAL_URL") + ":" + os.getenv("ES_LOCAL_PORT")
            )
            config["params_vectorbase"]["auth"] = [
                "elastic",
                os.getenv("ES_LOCAL_PASSWORD"),
            ]
        with open(file, "w") as f:
            json.dump(config, f, indent=4)


if "rag" not in st.session_state:
    st.session_state["rag_name"] = "naive"

if "api_key" not in st.session_state:
    st.session_state["api_key"] = st.session_state["config_server"]["params_host_llm"][
        "api_key"
    ]

if "all_rags" not in st.session_state:
    with open("streamlit_/utils/all_rags.json", "r") as file:
        all_rags = json.load(file)
    st.session_state["all_rags"] = all_rags

if "benchmark" not in st.session_state:
    st.session_state["benchmark"] = {}
    list_rags = list(st.session_state["all_rags"]["vllm"].keys())
    list_rags += list(st.session_state["all_rags"]["ollama"].keys())
    list_rags += list(st.session_state["all_rags"]["openai"].keys())
    list_rags += list(st.session_state["all_rags"]["mistral"].keys())
    list_rags = set(list_rags)
    st.session_state["benchmark"]["rags"] = dict(
        zip(
            list_rags,
            [False for i in range(len(list_rags))],
        )
    )
    st.session_state["benchmark"]["queries"] = pd.DataFrame(
        data={"query": [], "answer": []}
    )
    st.session_state["benchmark"]["load"] = False

if "custom_rags" not in st.session_state:
    if not os.path.exists("data/custom_rags/vllm"):
        os.makedirs("data/custom_rags/vllm")
    if not os.path.exists("data/custom_rags/ollama_openai_mistral"):
        os.makedirs("data/custom_rags/ollama_openai_mistral")

    custom_rags = os.listdir("data/custom_rags/vllm")
    custom_rags += os.listdir("data/custom_rags/ollama_openai_mistral")

    custom_rags = [
        custom_rag[:-5] for custom_rag in custom_rags if custom_rag != ".gitkeep"
    ]

    st.session_state["custom_rags"] = custom_rags

    if ".gitkeep" in st.session_state["custom_rags"]:
        st.session_state["custom_rags"].remove(".gitkeep")
if "databases" not in st.session_state:
    st.session_state["databases"] = {}

if "success" not in st.session_state:
    st.session_state["success"] = False

if "all_databases" not in st.session_state:
    if not os.path.exists("./data/databases"):
        os.makedirs("./data/databases")

    all_db = os.listdir("./data/databases")
    if ".gitkeep" in all_db:
        all_db.remove(".gitkeep")
    st.session_state["all_databases"] = all_db
    if "chat_database_name" not in st.session_state:
        st.session_state["chat_database_name"] = all_db[0] if len(all_db) > 0 else None
    if "benchmark_database" not in st.session_state:
        st.session_state["benchmark_database"] = all_db[0] if len(all_db) > 0 else None

for k, v in st.session_state.items():
    st.session_state[k] = v

pg.run()
