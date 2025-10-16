from typing import Union
import numpy as np
from elasticsearch import Elasticsearch, helpers
from .base_classes import VectorBase
from .utils_vlm import load_element
from .agent import Agent

from ..database.rag_classes import Chunk
from sqlalchemy.orm import DeclarativeMeta
from sqlalchemy import Integer, String
from typing import Type


def get_mapping_vb_embedding(class_: Type[DeclarativeMeta], vector_dims: int) -> dict:
    """
    Generate an Elastic search mapping
    """
    es_properties = {}

    for column in class_.__table__.columns:
        # Mapping SQLAlchemy -> Elasticsearch
        if isinstance(column.type, Integer):
            es_properties[column.name] = {"type": "integer"}
        elif isinstance(column.type, String):
            es_properties[column.name] = {"type": "text"}
        else:

            es_properties[column.name] = {"type": "text"}

    es_properties["vector"] = {
        "type": "dense_vector",
        "dims": vector_dims,
        "index": True,
        "similarity": "cosine",
    }

    return {"mappings": {"properties": es_properties}}


def get_mapping_to_bm25(
    class_: Type[DeclarativeMeta],
    k1: float = 1.2,
    b: float = 0.75,
) -> dict:
    """
    Generate an Elastic search mapping adapted to BM25
    """

    es_properties = {}

    for column in class_.__table__.columns:
        if isinstance(column.type, Integer):
            es_properties[column.name] = {"type": "integer"}
        elif isinstance(column.type, String):
            es_properties[column.name] = {"type": "text", "similarity": "default"}
        else:
            es_properties[column.name] = {"type": "keyword"}

    mapping = {
        "settings": {
            "similarity": {
                "default": {
                    "type": "BM25",
                    "k1": k1,
                    "b": b,
                }
            }
        },
        "mappings": {"properties": es_properties},
    }

    return mapping


def chunk_to_dict_vb_embedding(chunk, embedding):
    """
    Convert un Chunk object into a dict ready to be added to the vectorbase VectorBase_embeddings_elasticsearch
    """
    data = {}

    for col in chunk.__table__.columns:
        data[col.name] = getattr(chunk, col.name)

    data["vector"] = embedding

    return data


def chunk_to_dict_vb_bm25(chunk):
    """
    Convert a  Chunk into a dict ready to be added to the vectorbase BM25
    """
    data = {}

    for col in chunk.__table__.columns:
        data[col.name] = getattr(chunk, col.name)

    return data


def reconstruct_chunk_after_k_search(
    k_search_output: dict, class_: Type[DeclarativeMeta]
) -> object:

    valid_columns = {col.name for col in class_.__table__.columns}

    filtered_source = {k: v for k, v in k_search_output.items() if k in valid_columns}

    return class_(**filtered_source)


class Multiple_Vectorbase:

    def __init__(self):
        self.vectorbase = {}


class VectorBase_embeddings_elasticsearch(VectorBase):
    def __init__(
        self,
        vb_name: str,
        url_elasticsearch: str,
        agent,
        auth,
        embedding_model: str,
    ):

        self.embedding_model = embedding_model
        self.url_elasticsearch = url_elasticsearch
        self.vb_name = vb_name
        self.agent = agent
        self.auth = auth

        self.dimension = len(
            self.agent.embeddings("test", model=embedding_model)["embeddings"][0]
        )

        self.nb_tokens_embeddings = 0
        self.build_connection()

    def build_connection(self):
        self.client = Elasticsearch(
            self.url_elasticsearch,
            basic_auth=self.auth,
            request_timeout=60,
            max_retries=3,
            retry_on_timeout=True,
        )

    def create_collection(self, name=None, add_fields=[]) -> None:
        if name is None:
            name = self.vb_name

        mapping = get_mapping_vb_embedding(class_=Chunk, vector_dims=self.dimension)

        if len(add_fields) > 0:
            for field in add_fields:
                if field["data"]["datatype"] == "str":
                    field["data"]["datatype"] = "text"
                mapping["mappings"]["properties"][field["field_name"]] = {
                    "type": field["data"]["datatype"]
                }

        if not self.check_collection_exist(collection_name=name):
            self.client.indices.create(index=name, body=mapping)

        else:
            print(f'The collection "{name}" already exists')

    def check_collection_exist(self, collection_name):
        print(f"[DEBUG] Checking collection: {collection_name}")
        return self.client.indices.exists(index=collection_name)

    def check_element_exist(self, element, collection_name=None):
        if collection_name is None:
            collection_name = self.vb_name

        query = {"query": {"match_phrase": {"text": element}}}
        response = self.client.search(index=collection_name, body=query)

        if len(response["hits"]["hits"]) > 0:
            return True
        else:
            return False

    def add_str_batch_elements(
        self,
        chunks: list[Chunk],
        display_message: bool = True,
        collection_name=None,
    ) -> None:
        if collection_name is None:
            collection_name = self.vb_name

        if not self.check_collection_exist(collection_name=collection_name):
            self.create_collection(name=collection_name)

        data = []
        if chunks != []:
            texts = [chunk.text for chunk in chunks]

            embeddings = self.agent.embeddings(texts=texts, model=self.embedding_model)
            nb_embeddings_tokens = embeddings["nb_tokens"]
            if type(nb_embeddings_tokens) is list:
                nb_embeddings_tokens = np.sum(nb_embeddings_tokens)

        for k, chunk in enumerate(chunks):
            source = chunk_to_dict_vb_embedding(
                chunk=chunk, embedding=embeddings["embeddings"][k]
            )

            temp = {
                "_index": collection_name,
                "_source": source,
            }

            data.append(temp)
        res = helpers.bulk(self.client, data)
        if display_message:
            print(
                f"{len(data)} elements have been successfuly added in the vector base"
            )
        else:
            if display_message:
                print(
                    f"All the elements already were in the collection {collection_name}"
                )
        self.nb_tokens_embeddings += nb_embeddings_tokens
        return nb_embeddings_tokens

    def add_str_elements(
        self,
        chunks=list[Chunk],
        display_message: bool = True,
        collection_name=None,
    ) -> None:
        if collection_name is None:
            collection_name = self.vb_name

        if not chunks:
            if display_message:
                print("No chunks to add.")
            return

        if not self.check_collection_exist(collection_name=collection_name):
            self.create_collection(name=collection_name)

        data = []
        if chunks != []:
            texts = [chunk.text for chunk in chunks]
            embeddings = self.agent.embeddings(texts=texts, model=self.embedding_model)
            nb_embeddings_tokens = embeddings["nb_tokens"]
            if type(nb_embeddings_tokens) is list:
                nb_embeddings_tokens = np.sum(nb_embeddings_tokens)

            for k, chunk in enumerate(chunks):
                data = chunk_to_dict_vb_embedding(
                    chunk=chunk, embedding=embeddings["embeddings"][k]
                )

                res = self.client.index(index=collection_name, document=data)

            if display_message:
                print(
                    f"{len(data)} elements have been successfuly added to the vector base"
                )

        else:
            if display_message:
                print(
                    f"All the elements already were in the collection {collection_name}"
                )
        self.nb_tokens_embeddings += nb_embeddings_tokens
        return nb_embeddings_tokens

    def k_search(
        self,
        queries: Union[str, list[str]],
        k: int,
        output_fields: list[str] = ["text"],
        filters: dict = None,
        collection_name=None,
        type_output = Chunk
    ):

        if collection_name is None:
            collection_name = self.vb_name
        data = self.agent.embeddings(texts=queries, model=self.embedding_model)

        res = []
        num_candidates = max(500, k*5)
        for i in range(len(data["embeddings"])):
            body = {
                "size": k,
                "knn": {
                    "field": "vector",
                    "query_vector": data["embeddings"][i],
                    "k": k,  
                    "num_candidates": num_candidates, 
                },
            }

            response = self.client.search(index=collection_name, body=body)
            res.append(response)
        results = []
        for l in range(len(res)):
            result = []
            for i in range(np.min([len(res[l]["hits"]["hits"]), k])):
                source = res[l]["hits"]["hits"][i]["_source"]
                result.append(reconstruct_chunk_after_k_search(source, type_output))
            results.append(result)
        return results

    def delete_collection(self, vb_name = None):
        if vb_name is None:
            vb_name = self.vb_name
        if not self.check_collection_exist(vb_name):
            print("The collection does not exist")

        try:
            self.client.indices.delete(index=vb_name)
            print("the collection have been deleted")
        except Exception as e:
            print(f"Error while deleting the collection : {e}")


class VectorBase_BM25_elasticsearch(VectorBase):
    def __init__(self, vb_name: str, url_elasticsearch: str, auth):

        self.url_elasticsearch = url_elasticsearch
        self.vb_name = vb_name
        self.auth = auth
        self.nb_tokens_embeddings = 0

        self.build_connection()

    def build_connection(self):
        self.client = Elasticsearch(
            self.url_elasticsearch,
            basic_auth=self.auth,
            request_timeout=60,
            max_retries=3,
            retry_on_timeout=True,
        )

    def create_collection(self, name=None, add_fields=[]) -> None:
        if name is None:
            name = self.vb_name

        mapping = get_mapping_to_bm25(class_=Chunk)

        if len(add_fields) > 0:
            for field in add_fields:
                if field["data"]["datatype"] == "str":
                    field["data"]["datatype"] = "text"
                mapping["mappings"]["properties"][field["field_name"]] = {
                    "type": field["data"]["datatype"]
                }

        if not self.check_collection_exist(collection_name=name):
            self.client.indices.create(index=name, body=mapping)
        else:
            print(f'The collection "{name}" already exists')

    def check_collection_exist(self, collection_name):
        return self.client.indices.exists(index=collection_name)

    def check_element_exist(self, element, collection_name=None):
        if collection_name is None:
            collection_name = self.vb_name

        query = {"query": {"match_phrase": {"text": element}}}
        response = self.client.search(index=collection_name, body=query)

        if len(response["hits"]["hits"]) > 0:
            return True
        else:
            return False

    def add_str_batch_elements(
        self,
        chunks=list[Chunk],
        display_message: bool = True,
        collection_name=None,
    ) -> None:
        nb_embedding_tokens = 0
        if collection_name is None:
            collection_name = self.vb_name

        if not self.check_collection_exist(collection_name=collection_name):
            self.create_collection(name=collection_name)

        data = []
        if chunks != []:
            for k, chunk in enumerate(chunks):

                source = chunk_to_dict_vb_bm25(chunk=chunk)

                temp = {
                    "_index": collection_name,
                    "_source": source,
                }

                data.append(temp)
            helpers.bulk(self.client, data)
            if display_message:
                print(
                    f"{len(data)} elements have been successfuly added to the vector base"
                )
        else:
            if display_message:
                print(
                    f"All the elements already were in the collection {collection_name}"
                )
        return nb_embedding_tokens

    def add_str_elements(
        self,
        chunks=list[Chunk],
        display_message: bool = True,
        collection_name=None,
    ) -> None:
        nb_embedding_tokens = 0
        if collection_name is None:
            collection_name = self.vb_name

        if not self.check_collection_exist(collection_name=collection_name):
            self.create_collection(name=collection_name)

        data = []
        if chunks != []:
            for k, chunk in enumerate(chunks):
                data = chunk_to_dict_vb_bm25(chunk)

            self.client.index(index=collection_name, document=data)

            if display_message:
                print(
                    f"{len(data)} elements have been successfuly added to the vector base"
                )
        else:
            if display_message:
                print(
                    f"All the elements already were in the collection {collection_name}"
                )
        return nb_embedding_tokens

    def k_search(
        self,
        queries: Union[str, list[str]],
        k: int,
        output_fields: list[str] = ["text"],
        filters: dict = None,
        collection_name=None,
        type_output = Chunk
    ):
        # if type(queries) is str:
        #     queries = [queries]
        if collection_name is None:
            collection_name = self.vb_name
        res = []
        for i in range(len(queries)):
            response = self.client.search(
                index=collection_name,
                body={"size": k, "query": {"match": {"text": queries[i]}}},
            )
            res.append(response)

        results = []
        for l in range(len(res)):
            result = []
            for i in range(np.min([len(res[l]["hits"]["hits"]), k])):
                source = res[l]["hits"]["hits"][i]["_source"]
                result.append(reconstruct_chunk_after_k_search(source, type_output))
            results.append(result)
        return results

    def delete_collection(self, vb_name = None):
        if vb_name is None:
            vb_name = self.vb_name
        if not self.check_collection_exist(vb_name):
            print("The collection does not exist")

        try:
            self.client.indices.delete(index=vb_name)
            print("the collection have been deleted")
        except Exception as e:
            print(f"Error while deleting the collection : {e}")


class VectorBase_hybrid_elasticsearch(VectorBase_embeddings_elasticsearch):
    def __init__(
        self, vb_name: str, url_elasticsearch: str, agent, auth, embedding_model: str
    ):

        super().__init__(
            vb_name=vb_name,
            url_elasticsearch=url_elasticsearch,
            embedding_model=embedding_model,
            agent=agent,
            auth=auth,
        )

    def k_search(
        self,
        queries: Union[str, list[str]],
        k: int,
        output_fields: list[str] = ["text"],
        filters: dict = None,
        collection_name=None,
        type_output = Chunk
    ) -> list[Chunk]:

        if type(queries) is type(""):
            queries = [queries]
        if collection_name is None:
            collection_name = self.vb_name

        data = self.agent.embeddings(texts=queries,
                                     model=self.embedding_model)
        res = []
        num_candidates = max(500, k*5)
        for i in range(len(queries)):
            response = self.client.search(
                index=collection_name,
                body={
                    "size": k,
                    "query": {
                        "script_score": {
                            "query": {
                                "bool": {
                                    "should": [
                                        {
                                            "match": {"text": queries[i]}
                                        },  # Recherche BM25
                                        {
                                            "knn": {
                                                "field": "vector",
                                                "query_vector": data["embeddings"][i],
                                                "k": k,
                                                "num_candidates": num_candidates,
                                            }
                                        },  # Recherche Vectorielle
                                    ]
                                }
                            },
                            "script": {
                                "source": "0.5 * _score + 0.5 * cosineSimilarity(params.query_vector, 'vector')",
                                "params": {"query_vector": data["embeddings"][i]},
                            },
                        }
                    },
                },
            )
            res.append(response)
        results = []
        for l in range(len(res)):
            result = []
            # print("res", res[l]["hits"]["hits"], len(res[l]["hits"]["hits"]))
            for i in range(np.min([len(res[l]["hits"]["hits"]), k])):
                source = res[l]["hits"]["hits"][i]["_source"]
                result.append(reconstruct_chunk_after_k_search(source, type_output))
            results.append(result)
        return results

    def delete_collection(self, vb_name = None):
        if vb_name is None:
            vb_name = self.vb_name
        if not self.check_collection_exist(vb_name):
            print("The collection does not exist")

        try:
            self.client.indices.delete(index=vb_name)
            print("the collection have been deleted")
        except Exception as e:
            print(f"Error while deleting the collection : {e}")

class VectorBaseVlm_elasticsearch:
    def __init__(
        self, vb_name: str, url_elasticsearch: str, embedding_model, agent, auth
    ):

        self.vb_name = vb_name
        self.auth = auth
        self.agent = agent
        self.url_elasticsearch = url_elasticsearch
        self.embedding_model = embedding_model
        self.dimension = len(
            self.agent.embeddings_vlm(
                queries="test", model=embedding_model, mode="vlm"
            )["embeddings"][0]
        )
        self.client = Elasticsearch(self.url_elasticsearch, basic_auth=auth)

    def get_current_doc_id(self, collection_name=None):
        if collection_name is None:
            collection_name = self.vb_name
        if not self.check_collection_exist(collection_name=collection_name):
            return 0

        response = self.client.search(
            index=collection_name,
            body={
                "size": 0,
                "_source": False,
                "aggs": {"max_doc_id": {"max": {"field": "doc_id"}}},
            },
        )

        return response["aggregations"]["max_doc_id"]["value"]

    def create_collection(self, name=None) -> None:
        if name is None:
            name = self.vb_name

        mapping = {
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": self.dimension,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "id": {"type": "long"},
                    "doc_id": {"type": "long"},
                    "path": {"type": "text"},
                }
            }
        }
        if not self.check_collection_exist(collection_name=name):
            self.client.indices.create(index=name, body=mapping)
        else:
            print(f'The collection "{name}" already exists')

    def check_collection_exist(self, collection_name):
        return self.client.indices.exists(index=collection_name)

    def check_element_exist(self, element, collection_name=None):
        if collection_name is None:
            collection_name = self.vb_name

        query = {"query": {"match_phrase": {"path": element}}}
        response = self.client.search(index=collection_name, body=query)

        if len(response["hits"]["hits"]) > 0:
            return True
        else:
            return False

    def add_str_elements(
        self,
        elements: list[str],
        doc_id: int,
        metadata: dict = None,
        display_message: bool = True,
        collection_name=None,
    ) -> None:

        if collection_name is None:
            collection_name = self.vb_name

        if not self.check_collection_exist(collection_name=collection_name):
            self.create_collection(name=collection_name)

        filtered_elements = [
            element
            for element in elements
            if not self.check_element_exist(
                element=element, collection_name=collection_name
            )
        ]

        data = []
        if filtered_elements != []:
            images = []
            for i in range(len(filtered_elements)):
                image, is_empty = load_element(filtered_elements[i])
                if not is_empty:
                    images.append(image)
            if len(images) > 0:
                embeddings = self.agent.embeddings_vlm(
                    images=images, model=self.embedding_model, mode="vlm"
                )["embeddings"]

                data = {
                    "doc_id": doc_id,
                    "vector": embeddings[i],
                    "path": filtered_elements[i],
                }
                res = self.client.index(index=collection_name, document=data)

            if display_message:
                print(
                    f"{len(data)} elements have been successfuly added in the vector base"
                )
        else:
            if display_message:
                print(
                    f"All the elements already were in the collection {collection_name}"
                )

    def k_search(
        self,
        queries: Union[str, list[str]],
        k: int,
        output_fields: list[str] = ["text"],
        filters: dict = None,
        collection_name=None,
    ):

        if collection_name is None:
            collection_name = self.vb_name

        if type(queries) is str:
            queries = [queries]

        data = self.agent.embeddings_vlm(
            queries=queries, model=self.embedding_model, mode="vlm"
        )

        res = []
        num_candidates = max(500, k*5)
        for i in range(len(data["embeddings"])):
            response = self.client.search(
                index=collection_name,
                body={
                    "size": k,
                    "knn": {
                        "field": "vector",
                        "query_vector": data["embeddings"][i],
                        "k": k,  # Number of nearest neighbors to retrieve
                        "num_candidates": num_candidates,  # Number of candidates to consider
                    },
                },
            )
            res.append(response)

        results = []
        for l in range(len(res)):
            result = []
            for i in range(np.min([len(res[l]["hits"]["hits"]), k])):
                result.append(res[l]["hits"]["hits"][i]["_source"])
            results.append(result)
        return results

    def delete_collection(self):
        if not self.check_collection_exist(self.vb_name):
            print("The collection does not exist")

        try:
            self.client.indices.delete(index=self.vb_name)
            print("the collection have been deleted")
        except Exception as e:
            print(f"Error while deleting the collection : {e}")