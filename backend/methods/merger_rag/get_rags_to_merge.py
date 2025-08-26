from ..graph_rag.agent import GraphRagAgent

from ..naive_rag.agent import NaiveRagAgent
from ..agentic_rag.agent import AgenticRagAgent
from ..query_based_rag.agent import QueryBasedRagAgent
from ..self_rag.agent import SelfRagAgent
from ..corrective_rag.agent import CragAgent
from ..query_reformulation.agent import QueryReformulationRag
from backend.methods.reranker_rag.agent import RerankerRag
from ..semantic_chunking_rag.agent import SemanticChunkingRagAgent
from ..contextual_retrieval_rag.agent import ContextualRetrievalRagAgent
from ..advanced_rag.agent import AdvancedRag
from ..naive_chatbot.agent import NaiveChatbot
from ...utils.factory_name_dataser_vectorbase import get_name


def change_config_server(rag_name, config_server):
    if rag_name=="copali":
        config_server["type_retrieval"] = "vlm_embeddings"
        config_server["embedding_model"] = "vidore/colpali-v1.2-hf"
        config_server["model"] = "openbmb/MiniCPM-V-2_6"
    elif rag_name=="vlm":
        config_server["type_retrieval"] = "vlm_embeddings"
        config_server["embedding_model"] = "openbmb/VisRAG-Ret"
        config_server["model"] = "openbmb/MiniCPM-V-2_6"
    elif config_server["params_host_llm"]["type"] == "ollama":
        config_server["model"] = "gemma2:9b"
        config_server["embedding_model"] = "mxbai-embed-large:latest"
        
    elif config_server["params_host_llm"]["type"] == "vllm":
        config_server["model"] = "google/gemma-2-9b-it"
        config_server["embedding_model"] = "BAAI/bge-m3"
    elif config_server["params_host_llm"]["type"] == "openai":
        config_server["model"] = "gpt-4o-mini"
        config_server["embedding_model"] = "text-embedding-3-small"
    elif config_server["params_host_llm"]["type"] == "mistral":
        config_server["model"] = "mistral-small-latest"
        config_server["embedding_model"] = "mistral-embed"

    if rag_name!="copali" and rag_name!="vlm":
        if config_server["type_retrieval"] not in ["embeddings", "bm25", "hybrid"]:
            config_server["type_retrieval"] = "embeddings"
    return config_server


def get_rag_to_merge(rag_name, config_server, database_name=""):
    name = get_name(
        rag_name=rag_name,
        config_server=config_server,
        additionnal_name=database_name,
    )
    if rag_name == "naive":
        agent = NaiveRagAgent(config_server=config_server,
                              vb_name=name,
                              db_name=name)
    elif rag_name == "agentic":
        agent = AgenticRagAgent(config_server=config_server,
                              vb_name=name,
                              db_name=name)
        
    
        
    elif rag_name == "naive_chatbot":
        agent = NaiveChatbot(config_server=config_server)
    elif rag_name == "reranker_rag":
        agent = RerankerRag(config_server=config_server,
                            vb_name=name,
                            db_name=name)
    elif rag_name == "advanced_rag":
        agent = AdvancedRag(config_server=config_server, 
                            vb_name=name,
                            db_name=name)
    elif rag_name == "query_reformulation_rag":
        agent = QueryReformulationRag(config_server=config_server,
                                      vb_name=name,
                                      db_name=name)
    elif rag_name == "graph":
        agent = GraphRagAgent(config_server=config_server,
                              vb_name=name,
                              db_name=name)
    elif rag_name == "query_based":
        agent = QueryBasedRagAgent(
            config_server=config_server, vb_name=name, db_name=name
        )
    elif rag_name == "self":
        agent = SelfRagAgent(config_server=config_server, vb_name=name, db_name=name)
    elif rag_name == "crag":
        agent = CragAgent(config_server=config_server, vb_name=name, db_name=name)

    elif rag_name == "semantic_chunking":
        agent = SemanticChunkingRagAgent(
            config_server=config_server, vb_name=name, db_name=name
        )
    elif rag_name == "contextual_retrieval":
        agent = ContextualRetrievalRagAgent(
            config_server=config_server, vb_name=name, db_name=name
        )
    return agent


def get_custom_rag_agent(base_rag_name, config_server, database_name=""):
    name = (
        config_server["name"]
        + "_"
        + config_server["params_vectorbase"]["backend"]
        + "_"
        + database_name
    )
    # In order for the indexation to be done with the OpenAI embedding model
    if config_server["params_host_llm"]["type"] == "openai":
        name += "_openai"
    name = name.lower()
    if base_rag_name == "naive":
        agent = NaiveRagAgent(config_server=config_server, db_name=name, vb_name=name)
    elif base_rag_name == "reranker_rag":
        agent = RerankerRag(config_server=config_server, db_name=name, vb_name=name)
    elif base_rag_name == "advanced_rag":
        agent = AdvancedRag(
            config_server=config_server,
        )
    elif base_rag_name == "graph":
        agent = GraphRagAgent(config_server=config_server, vb_name=name, db_name=name)
    # elif base_rag_name=="multigraph":
    #     agent = MultiGraphRagAgent(model=config_server["model"],
    #                                storage_path=config_server["storage_path"],
    #                                language=config_server["language"],
    #                                params_host_llm=config_server["params_host_llm"],
    #                                embedding_model=config_server["embedding_model"],
    #                                params_vectorbase=config_server["params_vectorbase"])
    elif base_rag_name == "query_based":
        agent = QueryBasedRagAgent(
            config_server=config_server, db_name=name, vb_name=name
        )
    elif base_rag_name == "self":
        agent = SelfRagAgent(config_server=config_server, db_name=name, vb_name=name)
    elif base_rag_name == "crag":
        agent = CragAgent(config_server=config_server, db_name=name, vb_name=name)
    elif base_rag_name == "cag":
        agent = Cag(
            model=config_server["model"],
            storage_path=config_server["storage_path"],
            language=config_server["language"],
        )
    # elif base_rag_name=="vlm":
    #     agent = VLM_rag(model_params=config_server,
    #                     embedding_model=config_server["embedding_model"],
    #                     storage_path=config_server["storage_path"],
    #                     device=config_server["device"],
    #                     params_vectorbase=config_server["params_vectorbase"])
    # elif base_rag_name=="copali":
    #     agent = copali_rag(config=config_server,
    #                        storage_path=config_server["storage_path"],
    #                        embedding_model=config_server["embedding_model"],
    #                        device=config_server["device"])
    elif base_rag_name == "main":
        agent = MainRagAgent(config_server=config_server, db_name=name, vb_name=name)
    elif base_rag_name == "semantic_chunking":
        agent = SemanticChunkingRagAgent(
            config_server=config_server, db_name=name, vb_name=name
        )
    elif base_rag_name == "contextual_retrieval":
        agent = ContextualRetrievalRagAgent(
            config_server=config_server, db_name=name, vb_name=name
        )
    elif base_rag_name == "audio":
        agent = AudioRagAgent(config_server=config_server, db_name=name, vb_name=name)
    return agent
