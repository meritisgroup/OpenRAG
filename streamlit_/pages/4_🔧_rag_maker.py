import streamlit as st
import json
import re
import os
from elasticsearch import Elasticsearch
from pymilvus import utility, connections
import pandas as pd
from backend.utils.factory_name_dataser_vectorbase import get_name

config_new_rag = {}
st.markdown("# Customize your RAG:")


with open("streamlit_/utils/all_rags.json", "r") as file:
    base_rags = json.load(file)
base_rags = base_rags[st.session_state["config_server"]["params_host_llm"]["type"]]
del base_rags["naive_chatbot"]

config_new_rag["base"] = st.selectbox(
    label="**Choose your base RAG**",
    options=base_rags.keys(),
    format_func=lambda x: base_rags[x],
)

vectorbase_dict = {
    "elasticsearch": {
        "url": os.getenv("ES_LOCAL_URL") + ":" + os.getenv("ES_LOCAL_PORT"),
        "backend": "elasticsearch",
        "auth": ["elastic", os.getenv("ES_LOCAL_PASSWORD")],
        "batch": True,
    },
}

backend_vectorbase = {"elasticsearch": "Elastic Search"}

config_new_rag["params_vectorbase"] = vectorbase_dict["elasticsearch"]

retrieval_methods = {"embeddings": "Embeddings", "bm25": "BM25", "hybrid": "Hybrid"}
type_retrieval = st.selectbox(
    "**Choose retrieval method**",
    retrieval_methods.keys(),
    format_func=lambda x: retrieval_methods[x],
    index=list(retrieval_methods.keys()).index(
        st.session_state["config_server"]["type_retrieval"]
    ),
)

config_new_rag["type_retrieval"] = type_retrieval

splitter_dic = {
    "Semantic_TextSplitter": "Semantic Splitting",
    "Recursive_TextSplitter": "Recursive Splitting",
    "TextSplitter": "Length Splitting",
}
config_new_rag["TextSplitter"] = st.selectbox(
    label="**Choose text splitter**",
    options=splitter_dic.keys(),
    format_func=lambda x: splitter_dic[x],
)

config_new_rag["nb_chunks"] = st.slider(
    label="**Choose number of chunks to retrieve per query**",
    min_value=0,
    max_value=200,
    step=5,
    value=st.session_state["config_server"]["nb_chunks"],
    help="""The higher the number of value, the better the results of the RAG agent will be.
                                                           However a number of chunk too large might slow down the answer time and increase costs""",
)

st.markdown("## Chunk length")
if "chunk_length" not in st.session_state:
    st.session_state["chunk_length"] = st.session_state["config_server"]["chunk_length"]

if "numeric" not in st.session_state:
    st.session_state["numeric"] = st.session_state["config_server"]["chunk_length"]
    config_new_rag["chunk_length"] = st.session_state["numeric"]
if "indexing" not in st.session_state:
    st.session_state["indexing"] = False


def update_slider_from_num():
    st.session_state["chunk_length"] = st.session_state["numeric"]
    st.session_state["indexing"] = True


def update_num_from_slider():
    st.session_state["numeric"] = st.session_state["chunk_length"]
    st.session_state["indexing"] = True


st.number_input(
    "Chunk length",
    value=st.session_state["numeric"],
    key="numeric",
    on_change=update_slider_from_num,
    step=10,
)

st.slider(
    label="**Choose length of chunks for indexing phases:**",
    min_value=0,
    max_value=2000,
    step=10,
    value=st.session_state["chunk_length"],
    help=""" Keep in mind that too long or too short chunks can make the retrieval harder and decrease accuracy """,
    key="chunk_length",
    on_change=update_num_from_slider,
)

if "all_system_prompt" not in st.session_state:
    st.session_state["all_system_prompt"] = st.session_state["config_server"][
        "all_system_prompt"
    ]

system_prompt = st.selectbox(
    label="Choose the system prompt",
    options=st.session_state["all_system_prompt"].keys(),
)

config_new_rag["name"] = st.text_input(
    label="**Give your custom RAG a name:**",
    placeholder="Enter name",
)

if st.button(
    "**Create Custom RAG**",
    type="primary",
    help="Your RAG will only be visible in the LLM host at the moment of its creation, if you change LLM host, you must create your RAG again",
    use_container_width=True,
):

    if (
        config_new_rag["name"]
        in st.session_state["all_rags"][
            st.session_state["config_server"]["params_host_llm"]["type"]
        ]
    ):
        st.error(
            f"{config_new_rag['name']} already exists, please choose another name",
            icon="🚨",
        )

    elif not bool(re.fullmatch(r"^[a-z0-9_-]+$", config_new_rag["name"])):
        st.error(
            "Invalid RAG name, only alphanumeric characters, underscores, lowercase letters and hyphens are allowed",
            icon="🚨",
        )
    else:
        st.session_state["custom_rags"].append(config_new_rag["name"])

        local_params = {
            "generation_system_prompt_name": system_prompt,
            "forced_system_prompt": True,
        }
        saved_config = st.session_state["config_server"].copy()
        saved_config["base"] = config_new_rag["base"]
        saved_config["TextSplitter"] = config_new_rag["TextSplitter"]
        saved_config["type_retrieval"] = config_new_rag["type_retrieval"]
        saved_config["nb_chunks"] = config_new_rag["nb_chunks"]
        saved_config["name"] = config_new_rag["name"]
        saved_config["params_vectorbase"] = config_new_rag["params_vectorbase"]
        saved_config["local_params"] = local_params
        saved_config["chunk_length"] = st.session_state["chunk_length"]
        folder = (
            "vllm"
            if saved_config["params_host_llm"]["type"] == "vllm"
            else "ollama_openai_mistral"
        )
        with open(
            f"data/custom_rags/{folder}/{config_new_rag['name']}.json", "w"
        ) as config:
            json.dump(saved_config, config, ensure_ascii=False, indent=4)
        if folder == "vllm":
            st.session_state["all_rags"]["vllm"][config_new_rag["name"]] = (
                config_new_rag["name"]
            )
        else:
            st.session_state["all_rags"]["openai"][config_new_rag["name"]] = (
                config_new_rag["name"]
            )
            st.session_state["all_rags"]["ollama"][config_new_rag["name"]] = (
                config_new_rag["name"]
            )
            st.session_state["all_rags"]["mistral"][config_new_rag["name"]] = (
                config_new_rag["name"]
            )
        with open("streamlit_/utils/all_rags.json", "w") as file:
            json.dump(st.session_state["all_rags"], file, ensure_ascii=False, indent=4)
        st.session_state["benchmark"]["rags"][config_new_rag["name"]] = False
        st.success("RAG successfully created")


st.markdown("# Manage Custom RAGs:")

left, right = st.columns([0.85, 0.15], vertical_alignment="bottom")
rag_to_del = left.selectbox(
    label="List of custom RAGs",
    options=st.session_state["custom_rags"],
    label_visibility="collapsed",
)
if right.button(label="Delete RAG", type="primary", use_container_width=True):
    # Updating list of custom rag in streamlit and ./data/custom_rags folder
    st.session_state["custom_rags"].remove(rag_to_del)

    if (
        rag_to_del in st.session_state["all_rags"]["ollama"].keys()
        or rag_to_del in st.session_state["all_rags"]["openai"].keys()
        or rag_to_del in st.session_state["all_rags"]["mistral"].keys()
    ):
        del st.session_state["all_rags"]["ollama"][rag_to_del]
        del st.session_state["all_rags"]["openai"][rag_to_del]
        del st.session_state["all_rags"]["mistral"][rag_to_del]

    path = "./data/custom_rags/ollama_openai_mistral/" + rag_to_del + ".json"
    if os.path.exists(path):
        os.remove(path)

    if rag_to_del in st.session_state["all_rags"]["vllm"].keys():
        del st.session_state["all_rags"]["vllm"][rag_to_del]

    path = "./data/custom_rags/vllm/" + rag_to_del + ".json"
    if os.path.exists(path):
        os.remove(path)

    # Saving modified list of RAGs
    with open("streamlit_/utils/all_rags.json", "w") as file:
        json.dump(st.session_state["all_rags"], file, indent=4)

    # Deleting all indexation done with this RAG
    es = Elasticsearch(
        [st.session_state["config_server"]["params_vectorbase"]["url"]],
        basic_auth=(
            st.session_state["config_server"]["params_vectorbase"]["auth"][0],
            st.session_state["config_server"]["params_vectorbase"]["auth"][1],
        ),
    )
    for index_name in es.indices.get_alias(index="*"):
        if index_name.startswith(rag_to_del):
            es.indices.delete(index=index_name)
            print(f"{index_name} successfully deleted")

if rag_to_del in st.session_state["custom_rags"]:
    if rag_to_del in st.session_state["all_rags"]["vllm"]:
        path_folder = "vllm"
        folder = "vllm"
    else:
        path_folder = "ollama_openai_mistral"
        folder = "ollama"

    with open(f"data/custom_rags/{path_folder}/{rag_to_del}.json", "r") as file:
        config = json.load(file)
    retrieval_methods = {"embeddings": "Embeddings", "bm25": "BM25", "hybrid": "Hybrid"}
    display_config = {
        "Base RAG": [st.session_state["all_rags"][folder][config["base"]]],
        "Vectorbase Type": [backend_vectorbase[config["params_vectorbase"]["backend"]]],
        "Retrieval Method": [retrieval_methods[config["type_retrieval"]]],
        "Splitter": [splitter_dic[config["TextSplitter"]]],
        "Nb chunks retrieved": [str(config["nb_chunks"])],
    }

    st.write(pd.DataFrame(display_config))
    st.markdown("# Manage Indexations:")
if "rerun_managed_rag" not in st.session_state:
    st.session_state["rerun_managed_rag"] = False


def rerun_managed_rag():
    st.session_state["rerun_managed_rag"] = True
    st.session_state["indexation"] = None


st.selectbox(
    label="**Choose a RAG to view linked indexations**",
    options=base_rags.keys(),
    format_func=lambda x: base_rags[x],
    key="managed_rag",
    on_change=rerun_managed_rag,
)


rag_method = st.session_state["managed_rag"]

if (
    "custom_rags" in st.session_state.keys()
    and rag_method in st.session_state["custom_rags"]
):
    if st.session_state["config_server"]["params_host_llm"]["type"] == "vllm":
        folder = "vllm"
    else:
        folder = "ollama_openai_mistral"

    with open(f"data/custom_rags/{folder}/{rag_method}.json", "r") as file:
        custom_config = json.load(file)
    name = custom_config["name"]
    base = custom_config["base"]
else:
    name = get_name(rag_method, st.session_state["config_server"])
    pointer = name.find("_rag")
    name = name[:pointer]
    base = rag_method
list_indexation = []

es = Elasticsearch(
    [st.session_state["config_server"]["params_vectorbase"]["url"]],
    basic_auth=(
        st.session_state["config_server"]["params_vectorbase"]["auth"][0],
        st.session_state["config_server"]["params_vectorbase"]["auth"][1],
    ),
)
for index_name in es.indices.get_alias(index="*"):
    if index_name.startswith(name):
        list_indexation.append(index_name)

left, right = st.columns([0.85, 0.15])
st.session_state["indexation"] = left.selectbox(
    label="indexations", label_visibility="collapsed", options=list_indexation
)

if right.button(label="Delete indexation", use_container_width=True, type="primary"):

    if base == "graph":

        if "_local" in st.session_state.indexation:
            pointer = st.session_state.indexation.find("_local")
            suffix = "_global_search"
        elif "_global" in st.session_state.indexation:
            pointer = st.session_state.indexation.find("_global")
            suffix = "_local_search"

        if "milvus" in st.session_state.indexation:
            utility.drop_collection(st.session_state.indexation)
            utility.drop_collection(st.session_state.indexation[:pointer] + suffix)
        elif "elasticsearch" in st.session_state.indexation:
            es.indices.delete(index=st.session_state.indexation)
            es.indices.delete(index=st.session_state.indexation[:pointer] + suffix)

        for db in os.listdir("./storage"):
            if st.session_state.indexation[:pointer] in db:
                print("./storage/" + st.session_state.indexation[:pointer] + ".db")
                os.remove("./storage/" + st.session_state.indexation[:pointer] + ".db")
                st.session_state.indexation = None
                st.rerun()

    else:
        if "milvus" in st.session_state.indexation:
            utility.drop_collection(st.session_state.indexation)
        elif "elasticsearch" in st.session_state.indexation:
            es.indices.delete(index=st.session_state.indexation)

        for db in os.listdir("./storage"):
            if st.session_state.indexation in db:
                os.remove("./storage/" + st.session_state.indexation + ".db")
                st.session_state["rerun_managed_run"] = False
                st.session_state.indexation = None
                st.rerun()
