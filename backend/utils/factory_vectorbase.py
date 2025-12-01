from .base_classes import Agent
from .vectorbase_elasticsearch import (
    VectorBase_BM25_elasticsearch,
    VectorBase_embeddings_elasticsearch,
    VectorBase_hybrid_elasticsearch,
    VectorBaseVlm_elasticsearch,
)


def get_vectorbase(vb_name: str,
                   config_server: dict,
                   embedding_model: str,
                   agent: Agent = None):
    
    params_vectorbase = config_server["params_vectorbase"]
    type_retrieval = config_server["type_retrieval"]

    if params_vectorbase["backend"] == "elasticsearch":
        if type_retrieval == "embeddings":
            vb = VectorBase_embeddings_elasticsearch(
                vb_name=vb_name,
                url_elasticsearch=params_vectorbase["url"],
                embedding_model=embedding_model,
                agent=agent,
                auth=params_vectorbase["auth"],
            )
        elif type_retrieval == "bm25" or type_retrieval == "BM25":
            vb = VectorBase_BM25_elasticsearch(
                vb_name=vb_name,
                url_elasticsearch=params_vectorbase["url"],
                auth=params_vectorbase["auth"],
            )
        elif type_retrieval == "vlm_embeddings":
            vb = VectorBaseVlm_elasticsearch(
                vb_name=vb_name,
                url_elasticsearch=params_vectorbase["url"],
                embedding_model=embedding_model,
                agent=agent,
                auth=params_vectorbase["auth"],
            )
        elif type_retrieval == "hybrid":
            vb = VectorBase_hybrid_elasticsearch(
                vb_name=vb_name,
                url_elasticsearch=params_vectorbase["url"],
                embedding_model=embedding_model,
                agent=agent,
                auth=params_vectorbase["auth"],
            )

    return vb

