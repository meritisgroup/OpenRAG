import streamlit as st
from backend.factory_RagAgent import change_config_server
from streamlit_.utils.params_func import get_custom_rags_name, get_merge_rags_name
import json
import pandas as pd
import os
import glob
from ecologits import EcoLogits
from dotenv import load_dotenv

load_dotenv()

if not getattr(EcoLogits, "_initialized", False):
    EcoLogits.init()
    EcoLogits._initialized = True


chat = st.Page("streamlit_/pages/1_💬_chat.py", title="Chat")
config = st.Page("streamlit_/pages/2_🧠_configuration.py", title="Configuration")
benchmark = st.Page("streamlit_/pages/3_📚_benchmark.py", title="Benchmark")
rag_maker = st.Page("streamlit_/pages/4_🔧_rag_maker.py", title="Rag Maker")
databases = st.Page("streamlit_/pages/5_🌐_databases.py", title="Databases")
documentation = st.Page("streamlit_/pages/6_📖_documentation.py", title="Documentation")
advanced_configuration = st.Page(
    "streamlit_/pages/7_🛠️_advanced_configuration.py", title="Advanced Configuration"
)
metadatas = st.Page("streamlit_/pages/8_⚡_metadatas.py", title="Metadatas")


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
            metadatas,
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

if 'mode_interface' not in st.session_state:
    st.session_state.mode_interface = 'Simple' 

st.session_state.mode_interface = st.sidebar.radio(
    "Choisissez votre mode d'utilisation :",
    ['Simple', 'Avancé'],
    key='mode_inteface',
    horizontal=True,
)
st.sidebar.divider()

if "config_server" not in st.session_state:
    with open("data/base_config_server.json", "r") as file:
        base_config_server = json.load(file)
        st.session_state["config_server"] = base_config_server

    for file in glob.glob(f"data/custom_rags/*.json"):
        with open(file, "r") as f:
            config = json.load(f)
        with open(file, "w") as f:
            json.dump(config, f, indent=4)

if "rag" not in st.session_state:
    st.session_state["rag_name"] = "naive"

if "providers_infos" not in st.session_state:
    with open("data/providers_infos.json", "r") as file:
        providers_infos = json.load(file)
    st.session_state["providers_infos"] = providers_infos
    
if "api_key" not in st.session_state:
    provider_default_mode = st.session_state["config_server"]["default_mode_provider"]
    st.session_state["api_key"] = st.session_state["providers_infos"][provider_default_mode]["api_key"]


if "models_infos" not in st.session_state:
    with open("data/models_infos.json", "r") as file:
        models_infos = json.load(file)
    st.session_state["models_infos"] = models_infos

if "all_rags" not in st.session_state:
    with open("data/all_rags.json", "r") as file:
        models_infos = json.load(file)
    st.session_state["all_rags"] = models_infos


if "benchmark" not in st.session_state:
    st.session_state["benchmark"] = {}
    list_rags = list(st.session_state["all_rags"].keys())
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


st.session_state["custom_rags"] = get_custom_rags_name()
if ".gitkeep" in st.session_state["custom_rags"]:
    st.session_state["custom_rags"].remove(".gitkeep")

if "databases" not in st.session_state:
    st.session_state["databases"] = {}

if "merge_rags" not in st.session_state:
    if not os.path.exists(f"data/merge"):
        os.makedirs(f"data/merge")

    st.session_state["merge_rags"] = get_merge_rags_name()


if "rags_to_merge" not in st.session_state:
    st.session_state["rags_to_merge"] = {}
    list_rags = list(st.session_state["all_rags"].keys())
    list_rags = set(list_rags)
    st.session_state["rags_to_merge"]["rags"] = dict(
        zip(
            list_rags,
            [False for i in range(len(list_rags))],
        )
    )
    st.session_state["rags_to_merge"]["queries"] = pd.DataFrame(
        data={"query": [], "answer": []}
    )
    st.session_state["rags_to_merge"]["load"] = False


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

pg.run()
