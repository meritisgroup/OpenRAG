from .methods.graph_rag.agent import GraphRagAgent
from .methods.naive_rag.agent import NaiveRagAgent
from .methods.agentic_rag.agent import AgenticRagAgent
from .methods.merger_rag.agent import MergerRagAgent
from .methods.query_based_rag.agent import QueryBasedRagAgent
from .methods.self_rag.agent import SelfRagAgent
from .methods.corrective_rag.agent import CragAgent
from .methods.query_reformulation.agent import QueryReformulationRag
from backend.methods.reranker_rag.agent import RerankerRag
from .methods.semantic_chunking_rag.agent import SemanticChunkingRagAgent
from .methods.contextual_retrieval_rag.agent import ContextualRetrievalRagAgent
from .methods.advanced_rag.agent import AdvancedRag
from .methods.naive_chatbot.agent import NaiveChatbot

from .utils.factory_name_dataset_vectorbase import get_name

import json
import re



def change_config_server(rag_name, config_server):
    if rag_name == "copali":
        config_server["type_retrieval"] = "vlm_embeddings"
        config_server["embedding_model"] = "vidore/colpali-v1.2-hf"
        config_server["model"] = "openbmb/MiniCPM-V-2_6"
    elif rag_name == "vlm":
        config_server["type_retrieval"] = "vlm_embeddings"
        config_server["embedding_model"] = "openbmb/VisRAG-Ret"
        config_server["model"] = "openbmb/MiniCPM-V-2_6"
    elif config_server["params_host_llm"]["type"] == "ollama":
        config_server["model"] = "gemma3:12b"
        config_server["embedding_model"] = "mxbai-embed-large:latest"
        config_server["reranker_model"] = "gemma3:12b"
    elif config_server["params_host_llm"]["type"] == "vllm":
        config_server["model"] = "google/gemma-3-12b-it"
        config_server["embedding_model"] = "BAAI/bge-m3"
        config_server["reranker_model"] = "BAAI/bge-reranker-v2-m3"
    elif config_server["params_host_llm"]["type"] == "openai":
        config_server["model"] = "gpt-4o-mini"
        config_server["embedding_model"] = "text-embedding-3-small"
        config_server["reranker_model"] = "gpt-4.1-nano-2025-04-14"
        config_server["model_for_image"] = "gpt-4o-mini"
    elif config_server["params_host_llm"]["type"] == "mistral":
        config_server["model"] = "mistral-small-latest"
        config_server["embedding_model"] = "mistral-embed"
        config_server["reranker_model"] = "mistral-small-latest"

    if rag_name != "copali" and rag_name != "vlm":
        if config_server["type_retrieval"] not in ["embeddings", "bm25", "hybrid"]:
            config_server["type_retrieval"] = "embeddings"
    return config_server


def change_local_parameters(custom_local_config):
    config_path = "streamlit_/utils/base_config_server.json"

    with open(config_path, "r") as file:
        config = json.load(file)
        config["local_params"] = custom_local_config

    with open(config_path, "w") as file:
        json.dump(config, file, indent=4)


def put_default_local_parameters():
    default_local_config = {
        "forced_system_prompt": False,
        "generation_system_prompt_name": "default",
    }
    config_path = "streamlit_/utils/base_config_server.json"

    with open(config_path, "r") as file:
        config = json.load(file)
        config["local_params"] = default_local_config

    with open(config_path, "w") as file:
        json.dump(config, file, indent=4)


def get_rag_agent(rag_name, config_server, databases_name=[""]):
    names = [get_name(rag_name=rag_name,
                      config_server=config_server,
                      additionnal_name=databases_name[i]) for i in range(len(databases_name))]
    if rag_name == "naive":
        agent = NaiveRagAgent(config_server=config_server,
                             dbs_name=names,
                             data_folders_name=databases_name)
        
    elif rag_name == "agentic":
        agent = AgenticRagAgent(config_server=config_server,
                                dbs_name=names,
                                data_folders_name=databases_name)
    elif rag_name == "merger":
        agent = MergerRagAgent(config_server=config_server)
        
    elif rag_name == "naive_chatbot":
        agent = NaiveChatbot(config_server=config_server)
    elif rag_name == "reranker_rag":
        agent = RerankerRag(config_server=config_server,
                            dbs_name=names,
                            data_folders_name=databases_name)
    elif rag_name == "advanced_rag":
        agent = AdvancedRag(config_server=config_server,
                            dbs_name=names,
                            data_folders_name=databases_name)
    elif rag_name == "query_reformulation_rag":
        agent = QueryReformulationRag(
                            config_server=config_server,
                            dbs_name=names,
                            data_folders_name=databases_name)
    elif rag_name == "graph":
        agent = GraphRagAgent(config_server=config_server,
                              dbs_name=names,
                              data_folders_name=databases_name)
    elif rag_name == "query_based":
        agent = QueryBasedRagAgent(config_server=config_server,
                                  dbs_name=names,
                                  data_folders_name=databases_name)
    elif rag_name == "self":
        agent = SelfRagAgent(config_server=config_server,
                            dbs_name=names,
                            data_folders_name=databases_name)
    elif rag_name == "crag":
        agent = CragAgent(config_server=config_server, 
                          dbs_name=names,
                          data_folders_name=databases_name)

    elif rag_name == "semantic_chunking":
        agent = SemanticChunkingRagAgent(config_server=config_server,
                                         dbs_name=names,
                                         data_folders_name=databases_name)
        
    elif rag_name == "contextual_retrieval":
        agent = ContextualRetrievalRagAgent(config_server=config_server, 
                                            dbs_name=names,
                                            data_folders_name=databases_name)
    return agent



def get_custom_rag_agent(base_rag_name, config_server, databases_name=[""]):
    names = [config_server["name"] + "_" + config_server["params_vectorbase"]["backend"] + "_" + re.sub(r"[^a-zA-Z0-9]", "_", config_server["embedding_model"]) + "_" + "_" + config_server["params_host_llm"]["type"] + "_" + database_name for database_name in databases_name]
    
    names = [name.lower() for name in names]

    if base_rag_name == "naive":
        agent = NaiveRagAgent(config_server=config_server, 
                              dbs_name=names,
                              data_folders_name=databases_name)
    elif base_rag_name == "reranker_rag":
        agent = RerankerRag(config_server=config_server, 
                            dbs_name=names,
                            data_folders_name=databases_name)
    elif base_rag_name == "advanced_rag":
        agent = AdvancedRag(config_server=config_server, 
                            dbs_name=names,
                            data_folders_name=databases_name)
    elif base_rag_name == "merger":
        agent = MergerRagAgent(config_server=config_server, 
                              dbs_name=names,
                              data_folders_name=databases_name)
    elif base_rag_name == "graph":
        agent = GraphRagAgent(config_server=config_server, 
                              dbs_name=names,
                              data_folders_name=databases_name)
    # elif base_rag_name=="multigraph":
    #     agent = MultiGraphRagAgent(model=config_server["model"],
    #                                storage_path=config_server["storage_path"],
    #                                language=config_server["language"],
    #                                params_host_llm=config_server["params_host_llm"],
    #                                embedding_model=config_server["embedding_model"],
    #                                params_vectorbase=config_server["params_vectorbase"])
    elif base_rag_name == "query_based":
        agent = QueryBasedRagAgent(
            config_server=config_server, 
            dbs_name=names,
            data_folders_name=databases_name
        )
    elif base_rag_name == "self":
        agent = SelfRagAgent(config_server=config_server,
                             dbs_name=names,
                             data_folders_name=databases_name)
    elif base_rag_name == "crag":
        agent = CragAgent(config_server=config_server,
                          dbs_name=names,
                          data_folders_name=databases_name)
    elif base_rag_name == "semantic_chunking":
        agent = SemanticChunkingRagAgent(
            config_server=config_server, dbs_name=names, data_folders_name=databases_name
        )
    return agent
