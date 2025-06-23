from typing import Union
import numpy as np
from elasticsearch import Elasticsearch, helpers
from .base_classes import VectorBase
from .utils_vlm import load_element
from .agent import Agent


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
        self.client = Elasticsearch(self.url_elasticsearch,
                                    basic_auth=self.auth,
                                    request_timeout=60,   
                                    max_retries=3,
                                    retry_on_timeout=True)
        
    def create_collection(self, name=None, add_fields=[]) -> None:
        if name is None:
            name = self.vb_name

        mapping = {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": self.dimension,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "doc_name": {"type": "text"},
                }
            }
        }

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
        elements: list[str],
        docs_name: list[str] = None,
        metadata: list[dict] = [],
        display_message: bool = True,
        collection_name=None,
    ) -> None:
        if collection_name is None:
            collection_name = self.vb_name

        if not self.check_collection_exist(collection_name=collection_name):
            self.create_collection(name=collection_name)

        if docs_name is None:
            docs_name = []
            for k in range(len(elements)):
                docs_name.append("")

        data = []
        if elements != []:
            embeddings = self.agent.embeddings(
                texts=elements, model=self.embedding_model
            )
            nb_embeddings_tokens = embeddings["nb_tokens"]
            if type(nb_embeddings_tokens) is list:
                nb_embeddings_tokens = np.sum(nb_embeddings_tokens)

        for k, element in enumerate(elements):
            temp = {
                "_index": collection_name,
                "_source": {
                    "vector": embeddings["embeddings"][k],
                    "text": element,
                    "doc_name": docs_name[k],
                },
            }
            if len(metadata) > 0:
                for key in metadata[k].keys():
                    temp["_source"][key] = metadata[k][key]
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
        elements: list[str],
        docs_name: list[str] = None,
        metadata: list[dict] = [],
        display_message: bool = True,
        collection_name=None,
    ) -> None:
        if collection_name is None:
            collection_name = self.vb_name

        if not self.check_collection_exist(collection_name=collection_name):
            self.create_collection(name=collection_name)

        if docs_name is None:
            docs_name = []
            for k in range(len(elements)):
                docs_name.append("")

        data = []
        if elements != []:
            embeddings = self.agent.embeddings(
                texts=elements, model=self.embedding_model
            )
            nb_embeddings_tokens = embeddings["nb_tokens"]
            if type(nb_embeddings_tokens) is list:
                nb_embeddings_tokens = np.sum(nb_embeddings_tokens)

            for k, element in enumerate(elements):
                data = {
                    "vector": embeddings["embeddings"][k],
                    "text": element,
                    "doc_name": docs_name[k],
                }
                if len(metadata) > 0:
                    for key in metadata[k].keys():
                        data[key] = metadata[k][key]

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
    ):

        if collection_name is None:
            collection_name = self.vb_name
        data = self.agent.embeddings(texts=queries, model=self.embedding_model)

        res = []
        for i in range(len(data["embeddings"])):
            body = {
                "size": k,
                "knn": {
                    "field": "vector",
                    "query_vector": data["embeddings"][i],
                    "k": k,  # Number of nearest neighbors to retrieve
                    "num_candidates": 500,  # Number of candidates to consider
                },
            }
            response = self.client.search(index=collection_name, body=body)
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


class VectorBase_BM25_elasticsearch(VectorBase):
    def __init__(self, vb_name: str, url_elasticsearch: str, auth):

        self.url_elasticsearch = url_elasticsearch
        self.vb_name = vb_name
        self.auth = auth
        self.nb_tokens_embeddings = 0

        self.build_connection()

    def build_connection(self):
        self.client = Elasticsearch(self.url_elasticsearch,
                                    basic_auth=self.auth,
                                    request_timeout=60,   
                                    max_retries=3,
                                    retry_on_timeout=True)
        
    def create_collection(self, name=None, add_fields=[]) -> None:
        if name is None:
            name = self.vb_name

        mapping = {
            "settings": {
                "similarity": {"default": {"type": "BM25", "k1": 1.2, "b": 0.75}}
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text", "similarity": "default"},
                    "doc_name": {"type": "text", "similarity": "default"},
                }
            },
        }

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
        elements: list[str],
        docs_name: list[str] = None,
        metadata: list[dict] = [],
        display_message: bool = True,
        collection_name=None,
    ) -> None:
        nb_embedding_tokens = 0
        if collection_name is None:
            collection_name = self.vb_name

        if docs_name is None:
            docs_name = []
            for k in range(len(elements)):
                docs_name.append("")

        if not self.check_collection_exist(collection_name=collection_name):
            self.create_collection(name=collection_name)
        data = []
        if elements != []:
            for k, element in enumerate(elements):
                temp = {
                    "_index": collection_name,
                    "_source": {"text": element, "doc_name": docs_name[k]},
                }

                if len(metadata) > 0:
                    for key in metadata[k].keys():
                        temp["_source"][key] = metadata[k][key]
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
        elements: list[str],
        docs_name: list[str] = None,
        metadata: list[dict] = [],
        display_message: bool = True,
        collection_name=None,
    ) -> None:
        nb_embedding_tokens = 0
        if collection_name is None:
            collection_name = self.vb_name

        if not self.check_collection_exist(collection_name=collection_name):
            self.create_collection(name=collection_name)

        if docs_name is None:
            docs_name = []
            for k in range(len(elements)):
                docs_name.append("")

        data = []
        if elements != []:
            for k, element in enumerate(elements):
                data = {"text": element, "doc_name": docs_name[k]}
                if len(metadata) > 0:
                    for key in metadata[k].keys():
                        data[key] = metadata[k][key]
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
                result.append(res[l]["hits"]["hits"][i]["_source"])
            results.append(result)
        return results

    def delete_collection(self):
        if not self.check_collection_exist(self.vb_name):
            print("The collection does not exist")

        try:
            self.client.delete(index=self.vb_name)
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
    ):
        if type(queries) is type(""):
            queries = [queries]
        if collection_name is None:
            collection_name = self.vb_name

        data = self.agent.embeddings(texts=queries, model=self.embedding_model)
        res = []
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
                                                "num_candidates": 200,
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
        for i in range(len(data["embeddings"])):
            response = self.client.search(
                index=collection_name,
                body={
                    "size": k,
                    "knn": {
                        "field": "vector",
                        "query_vector": data["embeddings"][i],
                        "k": k,  # Number of nearest neighbors to retrieve
                        "num_candidates": 100,  # Number of candidates to consider
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
