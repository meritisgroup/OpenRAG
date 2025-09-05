

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
    elif rag_name == "agentic":
        name = "agentic_rag_{}_{}_{}".format(
            config_server["type_retrieval"],
            config_server["params_vectorbase"]["backend"],
            config_server["TextSplitter"],
        )
    elif rag_name == "merger":
        name = "merger_rag_{}_{}_{}".format(
            config_server["type_retrieval"],
            config_server["params_vectorbase"]["backend"],
            config_server["TextSplitter"],
        )

    name += "_{}".format(config_server["data_preprocessing"])
    name += "_{}".format(config_server["params_host_llm"]["type"])
    
    if additionnal_name != "":
        name += "_" + additionnal_name
    name = name.lower()
    return name
