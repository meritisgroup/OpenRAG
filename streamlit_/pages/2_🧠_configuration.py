import streamlit as st
import os
from backend.factory_RagAgent import get_rag_agent, change_config_server
from streamlit_.utils.chat_funcs import get_chat_agent
from backend.utils.utils_vlm import set_vllm_HF_key
from streamlit_.utils.params_func import get_custom_rags_name
import json
from urllib.parse import urlparse
from streamlit_.utils.params_func import modify_env


st.markdown("# Set Configuration")

# Setting params_host_llm
host_llm = {"vllm" : "vLLM", "ollama" : "Ollama",
            "openai" : "OpenAI", "mistral": "Mistral"}
def set_false():
    "When changin LLM host resests rags to perform benchmark on so that no unavailable Rags are called"
    for i in st.session_state["benchmark"]["rags"].keys():
        st.session_state["benchmark"]["rags"][i] = False

with open("data/providers_infos.json") as file:
    providers = json.load(file)
    host_dict = {
        "ollama": {"url": providers["ollama"]['url'], 
                "type": "ollama",
                "api_key": providers["ollama"]['api_key']},
        "vllm": {"url": providers["vllm"]['url'], 
                "type": "vllm",
                "api_key": providers["vllm"]['api_key']},
        "openai" : {"url" : providers["openai"]['url'], 
                    "type" : "openai", 
                    "api_key" : providers["openai"]['api_key']}
    }

if st.session_state["mode_interface"]=="Simple":
    llm_provider = st.session_state["config_server"]["default_mode_provider"]
    llm_provider = st.selectbox(
        label="**Choose LLM host:**",
        options=host_dict.keys(),
        format_func=lambda x:host_llm[x],
        on_change=set_false,
        index= list(host_dict.keys()).index(llm_provider),
        key="llm_host"
    )

    if llm_provider in ["ollama", "vllm"]:
        new_url = st.text_input(label=f"**{llm_provider} endpoint:**",
                                placeholder=host_dict[llm_provider]["url"],
                                disabled=False if llm_provider in ["ollama", "vllm"] else True,
                                key  = "endpoint")

    if llm_provider in ["openai"]:
        st.text_input(label="**OpenAI API Key:**",
                      value=host_dict['openai']["api_key"],
                      placeholder="Your API Key",
                      disabled=False if llm_provider in ["openai"] else True,
                      key  = "api_key",
                      type="password")

    st.text_input(label="**HuggingFace Token:**",
                  value=st.session_state["config_server"]["hf_token"],
                  placeholder="Mandatory for VLLM backend",
                  type="password",
                  key="hf_token")
    
else:
    st.subheader("📌 Configuration des API/modèles")
    with open("data/models_infos.json") as file:
        models_infos = json.load(file)

    model_names = list(models_infos.keys())
    model_names.append("➕ Ajouter un nouveau modèle")
    selected_model_name = st.selectbox("Sélectionner un modèle", model_names)

    if selected_model_name == "➕ Ajouter un nouveau modèle":
        new_model_name = st.text_input("Nom du modèle")
        new_model_url = st.text_input("URL du modèle")
        new_model_api_key = st.text_input("Clé API", type="password")
        
        # 🔽 Nouveau champ Type
        new_model_type = st.selectbox("Type du modèle", ["llm", "reranker", "embedding"])

        if st.button("Ajouter le modèle"):
            if new_model_name and new_model_url:
                if new_model_name in models_infos:
                    st.warning("Ce modèle existe déjà !")
                else:
                    models_infos[new_model_name] = {
                        "url": new_model_url,
                        "api_key": new_model_api_key,
                        "type": new_model_type  # ✅ Sauvegarde du type
                    }
                    with open("data/models_infos.json", "w") as file:
                        json.dump(models_infos, file, indent=4)
                    st.success(f"Modèle '{new_model_name}' ajouté ✅")
            else:
                st.warning("Merci de remplir tous les champs !")

    else:
        selected_model = models_infos[selected_model_name]
        with st.container():
            st.markdown(f"###### Modèle : {selected_model_name}")
            col1, col2 = st.columns([1, 2])
            with col1:
                new_model_type = st.selectbox(
                "Type du modèle", 
                ["llm", "reranker", "embedding"], 
                index=["llm", "reranker", "embedding"].index(selected_model.get("type", "llm"))
            )
                
            with col2:
                new_api_key = st.text_input("Clé API", value=selected_model.get('api_key', ''), type="password")

            new_url = st.text_input("URL", value=selected_model.get('url', ''))

        col_empty, col1_btn, col_empty1, col2_btn, col_empty2 = st.columns([0.5, 2, 0.5, 2, 0.5])
        with col1_btn:
            if st.button("💾 Sauvegarder les modifications", use_container_width=True):
                models_infos[selected_model_name]['url'] = new_url
                models_infos[selected_model_name]['api_key'] = new_api_key
                models_infos[selected_model_name]['type'] = new_model_type  

                with open("data/models_infos.json", "w") as file:
                    json.dump(models_infos, file, indent=4)
                st.session_state["models_infos"] = models_infos
                st.success("Modifications enregistrées ✅")

        with col2_btn:
            if st.button("🗑️ Supprimer le modèle", use_container_width=True):
                del models_infos[selected_model_name]
                with open("data/models_infos.json", "w") as file:
                    json.dump(models_infos, file, indent=4)
                st.success(f"Modèle '{selected_model_name}' supprimé ✅")

        st.markdown("<br><br>", unsafe_allow_html=True)

        roles = {
                "LLM de base": "model",
                "Reranker": "reranker_model",
                "Modèle pour décrire les images": "model_for_image",
                "Modèle d'embedding": "embedding_model"
                }
        task_mapping = {
                        "model": ["llm"],
                        "reranker_model": ["reranker", "llm"],
                        "model_for_image": ["llm"],
                        "embedding_model": ["embedding"]
                        }
        def sort_with_priority(model_name, config_key):
            model_type = models_infos[model_name]["type"]
            priorities = task_mapping.get(config_key, [])
            priority_index = priorities.index(model_type) if model_type in priorities else len(priorities)
            return (priority_index, model_name)

        config = {}
        st.subheader("📌 Configuration des modèles par rôle")

        for role_label, config_key in roles.items():
            
            if config_key not in st.session_state:
                st.session_state[config_key] = config.get(config_key, None)
            
            valid_tasks = task_mapping.get(config_key, [])
            filtered_models = [name for name, info in models_infos.items()
                            if info.get("type") in valid_tasks
                            ]
            options = [None] + sorted(filtered_models)

            col1, col2 = st.columns([0.5, 2])

            with col1:
                st.markdown(f"**Modèle pour {role_label}** {'✅' if st.session_state[config_key] else '❌'}")

            with col2:
                selected_model = st.selectbox(
                    label="", 
                    options=options,
                    index=options.index(st.session_state[config_key]),
                    format_func=lambda x: "Aucun modèle" if x is None else x,
                    key=f"model_select_{config_key}"
                )

            st.session_state[config_key] = selected_model
            config[config_key] = selected_model
            st.session_state["config_server"][config_key] = selected_model

        col_empty, col_save, col_empty1 = st.columns([1, 2, 1])

        st.markdown("<br>", unsafe_allow_html=True)
        with col_save:
            if st.button("💾 Sauvegarder les modèles par défaut", use_container_width=True):
                with open("data/base_config_server.json" , "w") as file:
                    json.dump(st.session_state["config_server"], file, indent=4)
                                
                st.success("✅ Modèles par défaut enregistrés !")

        st.markdown("<br><br>", unsafe_allow_html=True)













data_preparation = { "pdf_text_extraction":"PDF text extraction", 
                     "md_without_images": "PDF conversion into markdown" }


selected_data_prep = st.selectbox(label="**Choose data preparation method:**",
                                  options=list(data_preparation.keys()),
                                  format_func=lambda x:data_preparation[x],
                                  on_change=set_false,
                                  index= 0,
                                  key="data_prep")

# Setting params_vectorbase
def reset_retrieval():
    st.session_state["ret"] = "embeddings"

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

if "reformulate" not in st.session_state:
    st.session_state["reformulate"] = st.session_state["config_server"]["reformulate_query"]
st.toggle(
    "**Query reformulation ?**",
    help="Wether your query is reformulated by an LLM *before* being sent to the RAG",
    value= st.session_state["reformulate"],
    key="reformulate"
)
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
    if st.session_state["mode_interface"]=="Simple":
        if llm_provider in ["openai"]:
            host_dict[llm_provider]["api_key"] = st.session_state["api_key"]
        else:
            st.session_state["api_key"] = host_dict[llm_provider]["api_key"]

        st.session_state["config_server"]["default_mode_provider"] = llm_provider
        st.session_state["config_server"] = change_config_server(rag_name=None,
                                                                config_server=st.session_state["config_server"],
                                                                mode=st.session_state["mode_interface"])
        if type(st.session_state["config_server"]["hf_token"]) is str and len(st.session_state["config_server"]["hf_token"])>0:
            if st.session_state["config_server"]["default_mode_provider"]=="vllm":
                set_vllm_HF_key(url=host_dict[llm_provider]["url"],
                                key=st.session_state["config_server"]["hf_token"])

        if 'new_url' in globals() and new_url!="":
            host_dict[llm_provider]["url"] = new_url
            parsed = urlparse(new_url)

            base_url = f"{parsed.scheme}://{parsed.hostname}"
            port = parsed.port
            
            if llm_provider=="ollama":
                modify_env(key="ollama_LOCAL_URL",
                        value=base_url)
                modify_env(key="ollama_LOCAL_PORT",
                        value=port)
                host_dict[llm_provider]["url"] = os.getenv("ollama_LOCAL_URL")+":"+os.getenv("ollama_LOCAL_PORT")+"/v1"
            elif llm_provider=="vllm":
                modify_env(key="vllm_LOCAL_URL",
                        value=base_url)
                modify_env(key="vllm_LOCAL_PORT",
                        value=port)
                host_dict[llm_provider]["url"] = os.getenv("VLLM_LOCAL_URL")+":"+os.getenv("VLLM_LOCAL_PORT")
        st.session_state["config_server"]["default_mode_provider"] = llm_provider

        with open("data/providers_infos.json") as file:
            providers = json.load(file)
        
        providers[llm_provider] = {}
        providers[llm_provider]["url"] = host_dict[llm_provider]["url"]
        providers[llm_provider]["api_key"] = host_dict[llm_provider]["api_key"]

        with open("data/providers_infos.json" , "w") as file:
            json.dump(providers, file, indent=4)
        st.session_state["providers_infos"] = providers

        with open("data/models_infos.json") as file:
            models_infos = json.load(file)
        
        for key in ["model", "model_for_image", "embedding_model", "reranker_model"]:
            model = st.session_state["config_server"][key]
            if key not in models_infos.keys():
                models_infos[model] = {}
            models_infos[model]["url"] = host_dict[llm_provider]["url"]
            models_infos[model]["api_key"] = host_dict[llm_provider]["api_key"]

        with open("data/models_infos.json" , "w") as file:
            json.dump(models_infos, file, indent=4)
            
        st.session_state["models_infos"] = models_infos



    st.session_state["config_server"]["TextSplitter"] = st.session_state["split"]
    st.session_state["config_server"]["reformulate_query"] = st.session_state["reformulate"]
    st.session_state["config_server"]["type_retrieval"] = st.session_state["ret"]
    st.session_state["config_server"]["data_preprocessing"] = st.session_state["data_prep"]
    
    with open("data/base_config_server.json" , "w") as file:
        json.dump(st.session_state["config_server"], file, indent=4)

    st.session_state["custom_rags"] = get_custom_rags_name()
    rag_method = st.session_state["rag_name"]
    rag_agent = get_chat_agent(rag_method=rag_method,
                                databases_name=[])
    st.session_state["success"] = True
    st.session_state["rag"] = rag_agent
    st.session_state["rag_name"] = rag_method
