def get_name(rag_name, config_server, additionnal_name=""):
    name = ""
    if (
        rag_name == "naive"
        or rag_name == "crag"
        or rag_name == "reranker_rag"
        or rag_name == "query_reformulation_rag"
        or rag_name == "self"
        or rag_name == "main"
    ):
        name = "naive_rag_{}_{}_{}".format(
            config_server["type_retrieval"],
            config_server["params_vectorbase"]["backend"],
            config_server["TextSplitter"],
        )
    elif rag_name == "graph":
        name = "graph_rag_{}_{}_{}".format(
            config_server["type_retrieval"],
            config_server["params_vectorbase"]["backend"],
            config_server["TextSplitter"],
        )
    elif rag_name == "advanced_rag":
        name = "advanced_rag_{}_{}_{}".format(
            config_server["type_retrieval"],
            config_server["params_vectorbase"]["backend"],
            config_server["TextSplitter"],
        )
        if len(config_server["ProcessorChunks"]) > 0:
            for i in range(len(config_server["ProcessorChunks"])):
                name += "_{}".format(config_server["ProcessorChunks"][i])
    elif rag_name == "query_based":
        name = "query_rag_{}_{}_{}".format(
            config_server["type_retrieval"],
            config_server["params_vectorbase"]["backend"],
            config_server["TextSplitter"],
        )
    elif rag_name == "semantic_chunking":
        name = "semantic_rag_{}_{}_{}".format(
            config_server["type_retrieval"],
            config_server["params_vectorbase"]["backend"],
            config_server["TextSplitter"],
        )
    elif rag_name == "contextual_retrieval":
        name = "contextual_rag_{}_{}_{}".format(
            config_server["type_retrieval"],
            config_server["params_vectorbase"]["backend"],
            config_server["TextSplitter"],
        )
    elif rag_name == "audio":
        name = "audio_rag_{}".format(config_server["type_retrieval"])
    elif rag_name == "vlm":
        name = "vlm_rag"
    elif rag_name == "copali":
        name = "colpali_rag"

    if additionnal_name != "":
        name += "_" + additionnal_name

    if config_server["params_host_llm"]["type"] == "openai":
        name += "_openai"  # In order for the indexation to be done with the openai embedding model
    elif config_server["params_host_llm"]["type"] == "mistral":
        name += "_mistral"  # In order for the indexation to be done with the openai embedding model
    elif config_server["params_host_llm"]["type"] == "vllm":
        name += "_vllm"
    elif config_server["params_host_llm"]["type"] == "ollama":
        name += "_ollama"
    name = name.lower()
    return name
