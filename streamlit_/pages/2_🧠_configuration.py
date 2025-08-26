import streamlit as st
import os
from backend.factory_RagAgent import get_rag_agent, change_config_server
from streamlit_.utils.chat_funcs import get_chat_agent
from backend.utils.utils_vlm import set_vllm_HF_key
import json


st.markdown("# Set Configuration")

# Setting params_host_llm
host_llm = {"vllm" : "vLLM", "ollama" : "Ollama",
            "openai" : "OpenAI", "mistral": "Mistral"}
def set_false():
    "When changin LLM host resests rags to perform benchmark on so that no unavailable Rags are called"
    for i in st.session_state["benchmark"]["rags"].keys():
        st.session_state["benchmark"]["rags"][i] = False

host_llm = {"vllm" : "vLLM",
            "ollama" : "Ollama",
            "openai" : "OpenAI",
            "mistral" : "Mistral"}

host_dict = {
    "ollama": {"url": os.getenv("ollama_LOCAL_URL")+":"+os.getenv("ollama_LOCAL_PORT")+"/v1", 
               "type": "ollama",
               "api_key": None},
    "vllm": {"url": os.getenv("VLLM_LOCAL_URL")+":"+os.getenv("VLLM_LOCAL_PORT"), 
             "type": "vllm",
             "api_key": None},
    "openai" : {"url" : None, 
                "type" : "openai", 
                "api_key" : st.session_state["api_key"]},
    "mistral" : {"url" : None, 
                 "type" : "mistral", 
                 "api_key" : st.session_state["api_key"]}
}

if "llm_host" not in st.session_state:
    st.session_state["llm_host"] = st.session_state["config_server"]["params_host_llm"]["type"]

st.selectbox(
    label="**Choose LLM host:**",
    options=host_dict.keys(),
    format_func=lambda x:host_llm[x],
    on_change=set_false,
    index= list(host_dict.keys()).index(st.session_state["llm_host"]),
    key="llm_host"
)


st.text_input(
    label="**Mistral / OpenAI API Key:**",
    value=st.session_state["api_key"],
    placeholder="Your API Key",
    disabled=False if st.session_state["llm_host"] in ["openai", "mistral"] else True,
    key  = "api_key",
    type="password"
    )

st.session_state["config_server"]["params_host_llm"] = host_dict[st.session_state["llm_host"]]
st.text_input(
    label="**HuggingFace Token:**",
    value=st.session_state["hf_token"],
    placeholder="Mandatory for VLLM backend",
    type="password",
    key="hf_token")
st.session_state["config_server"]["hf_token"] = st.session_state.hf_token

# setting type of data preparation
data_preparation = { "pdf_text_extraction":"PDF text extraction", "md_without_images": "PDF conversion into markdown",
                     "md_with_images" : "PDF conversion into markdown with image description" }


selected_data_prep = st.selectbox(
    label="**Choose data preparation method:**",
    options=list(data_preparation.keys()),
    format_func=lambda x:data_preparation[x],
    on_change=set_false,
    index= 0,
    key="data_prep",
)

st.session_state["config_server"]["data_preprocessing"] = st.session_state["data_prep"]




# Setting params_vectorbase
def reset_retrieval():
    st.session_state["ret"] = "embeddings"

st.session_state["config_server"]["device"] = "cpu"

if st.session_state["config_server"]["params_host_llm"]["type"] == "ollama":
    st.session_state["config_server"]["model"] = "gemma2:9b"
    st.session_state["config_server"]["model_for_image"] = "gemma3:12b"
    st.session_state["config_server"]["embedding_model"] = "mxbai-embed-large:latest"

elif st.session_state["config_server"]["params_host_llm"]["type"] == "vllm":
    st.session_state["config_server"]["model"] = "google/gemma-2-9b-it"
    st.session_state["config_server"]["embedding_model"] = "BAAI/bge-m3"

elif st.session_state["config_server"]["params_host_llm"]["type"] == "openai":
    st.session_state["config_server"]["model"] = "o4-mini-2025-04-16"
    st.session_state["config_server"]["model_for_image"] = "o4-mini-2025-04-16"
    st.session_state["config_server"]["embedding_model"] = "text-embedding-3-small"

elif st.session_state["config_server"]["params_host_llm"]["type"] == "mistral":
    st.session_state["config_server"]["model"] = "mistral-small-latest"
    st.session_state["config_server"]["embedding_model"] = "mistral-embed"


st.session_state["config_server"]["reranker_model"] = "BAAI/bge-reranker-v2-m3"

if "ret" not in st.session_state:
    st.session_state["ret"] = st.session_state["config_server"]["type_retrieval"]

if st.session_state["config_server"]["params_vectorbase"]["backend"] == "milvus":
    retrieval_methods = {"embeddings" : "Embeddings"}
    st.selectbox(
        "**Choose retrieval method:**",
        retrieval_methods.keys(),
        format_func=lambda x: retrieval_methods[x],
        key = "ret"
    )
    
elif st.session_state["config_server"]["params_vectorbase"]["backend"] == "elasticsearch":
    retrieval_methods = {"embeddings" : "Embeddings", "bm25" : "BM25", "hybrid" : "Hybrid"}
    st.selectbox(
            "**Choose retrieval method:**",
            retrieval_methods.keys(),
            format_func=lambda x: retrieval_methods[x],
            index=list(retrieval_methods.keys()).index(st.session_state["ret"]),
            key = "ret"
        )

else:
    retrieval_methods = {"embeddings" : "Embeddings", "bm25" : "BM25", "hybrid" : "Hybrid"}
    st.selectbox(
        "**Choose retrieval method**",
        retrieval_methods.keys(),
        format_func=lambda x:retrieval_methods[x],
        index=list(retrieval_methods.keys()).index(st.session_state["ret"]),
        key="ret"
    )
st.session_state["config_server"]["type_retrieval"] = st.session_state["ret"]
languages = ["FR", "EN"]
if "lang" not in st.session_state:
    st.session_state["lang"] = st.session_state["config_server"]["language"]
st.selectbox(
    "**Choose RAG language:**",
    languages,
    index= languages.index(st.session_state["lang"]),
    key = "lang"
)
st.session_state["config_server"]["language"] = st.session_state["lang"]
splitter_dic = {
    "Semantic_TextSplitter": "Semantic Splitting",
    "Recursive_TextSplitter": "Recursive Splitting",
    "TextSplitter": "Length Splitting",
}
if "split" not in st.session_state:
    st.session_state["split"] = st.session_state["config_server"]["TextSplitter"]
st.selectbox(
    "**Choose TextSplitter:**",
    splitter_dic.keys(),
    help="""- **Semantic Splitting** : Spots semantic similarities between sentences and chunks accordingly \n
                                - **Recursive Splitting** : Divides text into smaller segments in a hierarchical and iterative manner, using a series of separators to preserve the structure and context of the text. \n
                                - **Length Splitting** : Splits text into chunks of fixed size
                                """,
    format_func=lambda x: splitter_dic[x],
    index=list(splitter_dic.keys()).index(st.session_state["split"]),
    key = "split"
)
st.session_state["config_server"]["TextSplitter"] = st.session_state["split"]

if "reformulate" not in st.session_state:
    st.session_state["reformulate"] = st.session_state["config_server"]["reformulate_query"]
st.toggle(
    "**Query reformulation ?**",
    help="Wether your query is reformulated by an LLM *before* being sent to the RAG",
    value= st.session_state["reformulate"],
    key="reformulate"
)
st.session_state["config_server"]["reformulate_query"] = st.session_state["reformulate"]

if "chunk" not in st.session_state:
    st.session_state["chunk"] = st.session_state["config_server"]["nb_chunks"]
st.slider(
    label="**Choose number of chunks to retrieve per query:**",
    min_value=0,
    max_value=200,
    step=5,
    value=st.session_state["chunk"],
    help="""The higher the number of value, the better the results of the RAG agent will be.
                                                           However a number of chunk too large might slow down the answer time and increase costs""",
    key = "chunk"
)

st.session_state["config_server"]["nb_chunks"] = st.session_state["chunk"]

if st.button("Save Configuration", type="primary", use_container_width=True):
    rag_method = st.session_state["rag_name"]
    rag_agent = get_chat_agent(rag_method=rag_method)
    st.session_state["success"] = True
    st.session_state["rag"] = rag_agent
    
    if type(st.session_state["config_server"]["hf_token"]) is str and len(st.session_state["config_server"]["hf_token"])>0:
        if st.session_state["config_server"]["params_host_llm"]["type"]=="vllm":
            set_vllm_HF_key(url=st.session_state["config_server"]["params_host_llm"]['url'],
                            key=st.session_state["config_server"]["hf_token"])

    st.session_state["rag_name"] = rag_method
    with open("streamlit_/utils/base_config_server.json" , "w") as file:
        json.dump(st.session_state["config_server"], file, indent=4)
