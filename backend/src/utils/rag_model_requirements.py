RAG_MODEL_REQUIREMENTS = {
    "naive": {"required": ["model", "embedding_model"], "optional": []},
    "naive_chatbot": {"required": ["model"], "optional": []},
    "advanced_rag": {
        "required": ["model", "embedding_model"],
        "optional": ["reranker_model"],
    },
    "agentic": {
        "required": ["model", "embedding_model", "reranker_model"],
        "optional": [],
    },
    "agentic_router": {
        "required": ["model", "embedding_model", "reranker_model"],
        "optional": [],
    },
    "reranker_rag": {
        "required": ["model", "embedding_model", "reranker_model"],
        "optional": [],
    },
    "query_reformulation_rag": {
        "required": ["model", "embedding_model"],
        "optional": [],
    },
    "semantic_chunking": {"required": ["model", "embedding_model"], "optional": []},
    "graph": {"required": ["model", "embedding_model"], "optional": []},
    "crag": {"required": ["model", "embedding_model"], "optional": []},
    "contextual_retrieval": {"required": ["model", "embedding_model"], "optional": []},
    "query_based": {"required": ["model", "embedding_model"], "optional": []},
    "self": {"required": ["model", "embedding_model"], "optional": []},
    "hyde": {"required": ["model", "embedding_model"], "optional": []},
    "merger": {"required": ["model"], "optional": []},
}


def get_required_models_for_rag(rag_name: str, config: dict) -> dict:
    import json
    import os

    custom_rag_path = f"data/custom_rags/{rag_name}.json"
    if os.path.exists(custom_rag_path):
        with open(custom_rag_path, "r") as f:
            custom_config = json.load(f)
        base_rag = custom_config.get("base", "naive")
        return get_required_models_for_rag(base_rag, config)

    merge_rag_path = f"data/merge/{rag_name}.json"
    if os.path.exists(merge_rag_path):
        return {"required": ["model"], "optional": []}

    requirements = RAG_MODEL_REQUIREMENTS.get(
        rag_name, {"required": ["model", "embedding_model"], "optional": []}
    )

    required = list(requirements["required"])

    type_retrieval = config.get("type_retrieval", "embeddings")
    if type_retrieval.lower() == "bm25" and "embedding_model" in required:
        required.remove("embedding_model")

    optional_models = []
    for opt_model in requirements["optional"]:
        if config.get(opt_model):
            optional_models.append(opt_model)

    return {"required": required, "optional": optional_models}
