import os

def get_possible_embeddings_model(provider):
    if provider=="ollama":
        return ["mxbai-embed-large:latest", "bge-m3:latest"]
    elif provider=="openai":
        return ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
    elif provider=="mistral":
        return ["mistral-embed"]
    elif provider=="vllm":
        return ["mixedbread-ai/mxbai-embed-large-v1", "BAAI/bge-m3", "Qwen/Qwen3-Embedding-0.6B"]
    

def get_default_embeddings_model(provider):
    return get_possible_embeddings_model(provider=provider)[0]

def get_custom_rags(provider=None):
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