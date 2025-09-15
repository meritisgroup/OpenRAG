import os
import json



def get_possible_embeddings_model(provider):
    if provider=="ollama":
        return ["mxbai-embed-large:latest", "bge-m3:latest", "all-minilm:22m"]
    elif provider=="openai":
        return ["text-embedding-3-small", "text-embedding-3-large",
                 "text-embedding-ada-002"]
    elif provider=="mistral":
        return ["mistral-embed"]
    elif provider=="vllm":
        return ["mixedbread-ai/mxbai-embed-large-v1", "BAAI/bge-m3",
                 "Qwen/Qwen3-Embedding-0.6B", "sentence-transformers/all-MiniLM-L6-v2"]
    

def get_default_embeddings_model(provider):
    return get_possible_embeddings_model(provider=provider)[0]


def get_config_rag(rag_name, provider):
    custom_rags_name = get_custom_rags_name(provider=provider)
    merge_rags_name = get_merge_rags_name(provider=provider)
    if rag_name in custom_rags_name:
        with open(f"data/custom_rags/{provider}/{rag_name}.json", "r") as file:
            config = json.load(file)
        return config
    elif rag_name in merge_rags_name:
        with open(f"data/merge/{provider}/{rag_name}.json", "r") as file:
            config = json.load(file)
        return config
    else:
        with open(f"streamlit_/utils/base_config_server.json", "r") as file:
            config = json.load(file)
        return config


def get_custom_rags_name(provider=None):
    custom_rags = []
    if provider is None:
        folders = ["vllm", "ollama", "openai", "mistral"]
    else:
        folders = [provider]
    for f in folders:
        if not os.path.exists(f"data/custom_rags/'{f}'"):
            os.makedirs(f"data/custom_rags/{f}", exist_ok=True)

        custom_rags += os.listdir(f"data/custom_rags/{f}")

    custom_rags = [
        custom_rag[:-5] for custom_rag in custom_rags if custom_rag != ".gitkeep"
    ]

    check_doublons = set()
    unique_rags = []
    for p in custom_rags:
        name = os.path.basename(p)
        if name not in check_doublons:
            check_doublons.add(name)
            unique_rags.append(p)

    return unique_rags


def get_merge_rags_name(provider=None):
    custom_rags = []
    if provider is None:
        folders = ["vllm", "ollama", "openai", "mistral"]
    else:
        folders = [provider]
    for f in folders:
        if not os.path.exists(f"data/merge/'{f}'"):
            os.makedirs(f"data/merge/{f}", exist_ok=True)

        custom_rags += os.listdir(f"data/merge/{f}")

    custom_rags = [
        custom_rag[:-5] for custom_rag in custom_rags if custom_rag != ".gitkeep"
    ]

    check_doublons = set()
    unique_rags = []
    for p in custom_rags:
        name = os.path.basename(p)
        if name not in check_doublons:
            check_doublons.add(name)
            unique_rags.append(p)

    return unique_rags