

def get_name(rag_name, config_server, additionnal_name=""):
    name = "{}_".format(config_server["embedding_model"])
    if (
        rag_name == "naive"
        or rag_name == "crag"
        or rag_name == "reranker_rag"
        or rag_name == "query_reformulation_rag"
        or rag_name == "self"
        or rag_name == "main"
    ):
        name = name + "{}_{}".format(
            config_server["type_retrieval"],
            config_server["TextSplitter"],
        )
    elif rag_name == "graph":
        name = "graph_rag_{}{}_{}".format(name,
                                          config_server["type_retrieval"],
                                          config_server["TextSplitter"])
    elif rag_name == "query_based":
        name = "query_rag_{}{}_{}".format(name,
                                          config_server["type_retrieval"],
                                          config_server["TextSplitter"],
                                        )
    elif rag_name == "advanced_rag":
        name = "{}{}_{}".format(name,
            config_server["type_retrieval"],
            config_server["TextSplitter"],
        )
        if len(config_server["ProcessorChunks"]) > 0:
            for i in range(len(config_server["ProcessorChunks"])):
                name += "_{}".format(config_server["ProcessorChunks"][i])

    elif rag_name == "semantic_chunking":
        name = "{}{}_{}".format(name,
            config_server["type_retrieval"],
            "Semantic_TextSplitter",
        )
    elif rag_name == "contextual_retrieval":
        name = "{}{}_{}".format(name,
            config_server["type_retrieval"],
            config_server["TextSplitter"],
            "Contextual"
        )

    name += "_{}".format(config_server["data_preprocessing"])
    if additionnal_name != "":
        name += "_" + additionnal_name
    name = name.lower()
    return name
