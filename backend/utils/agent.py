from .agent_functions import (
    predict,
    multiple_predict,
    predict_vllm,
    predict_json,
    predict_image,
    predict_mistral,
    multiple_predict_mistral,
    rerank,
    RerankedChunk,
)
from openai import OpenAI
from mistralai import Mistral
from jsonschema import validate
import json
import concurrent.futures
import requests
import time
import numpy as np
from typing import List
from .base_classes import Agent
from .agent_functions import np_array_to_file
from .threading_utils import get_executor_threads
from pydantic import BaseModel
from backend.database.rag_classes import Chunk


def get_Agent(config_server: dict, models_infos: dict):
    agent = Agent_openai(models_infos = models_infos,
                         language=config_server["language"],
                         max_attempts=config_server["max_attempts"],
                         max_workers=config_server["max_workers"])
    return agent


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



class Agent_Vllm(Agent):
    def __init__(
        self,
        models_infos,
        language="EN",
        max_attempts=5,
        max_workers = 10,
    ):  
        self.clients = {}
        for key in self.models_infos.keys():
            if "api_key" in self.models_infos[key].keys() and "url" in self.models_infos[key].keys():
                self.clients[key] = OpenAI(api_key=self.models_infos[key]["api_key"],
                                           base_url=self.models_infos[key]["url"])
                
        self.models_infos = models_infos
        self.language = language
        self.max_attempts = max_attempts
        self.temperature = 0
        self.max_workers = max_workers


    def multiple_predict(
        self,
        prompts: List[str],
        system_prompts: List[str],
        model: str,
        images = None,
        json_format=None,
        temperature=0,
        options_generation=None,
    ) -> str:
        """
        It formats the query with the good prompt, then gives this prompt to the LLM and return the cleaned output
        """

        url = self.url + "/predict"
        answers = predict_vllm(
            system_prompts=system_prompts,
            prompts=prompts,
            model=model,
            url=url,
            temperature=self.temperature,
            images=images,
            json_format=json_format,
            options_generation=options_generation,
        )

        return answers

    def predict(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        images: list[str] = None,
        temperature=0,
        options_generation=None,
    ) -> str:
        answer = self.multiple_predict(
            prompts=[prompt],
            system_prompts=[system_prompt],
            model = model,
            temperature=temperature,
            images=[images],
            options_generation=options_generation,
        )
        answer["texts"] = answer["texts"][0]
        return answer



    def multiple_predict_json(self, 
                              prompts: list[str],
                              system_prompts: list[str],
                              model : str,
                              json_format: BaseModel,
                              temperature=0, 
                              images: list[list[str]] = None,
                              options_generation = None):
        answers = self.multiple_predict(
            prompts=prompts,
            system_prompts=system_prompts,
            model = model,
            temperature=temperature,
            json_format=json_format,
            images=images,
            options_generation=options_generation,
        )
        results = []
        for i in range(len(answers["texts"])):
            answer_temp = answers["texts"][i]
            if "json" in answer_temp:
                cleaned = answer_temp.split("json\n")[1].split("\n```")[0]
            elif "{" in answer_temp:
                cleaned = answer_temp.split("{")[1].split("}")[0]
                cleaned = "{" + cleaned + "}"
            elif "True" in answer_temp and json_format.__name__ == "StatementSupported":
                cleaned = '{"supported": true}'

            elif "False" in answer_temp and json_format.__name__ == "StatementSupported":
                cleaned = '{"supported": false}'
            else:
                if images is not None and 0 <= i < len(images) and images[i] is not None: 
                    cleaned = self.multiple_predict_json(prompts = [prompts[i]],
                                                        system_prompts = [system_prompts[i]],
                                                        json_format = json_format,
                                                        model=model,
                                                        temperature = temperature+0.5 if temperature<1 else temperature, 
                                                        images = [images[i]],
                                                        options_generation = options_generation)
                else:
                    cleaned = self.multiple_predict_json(prompts = [prompts[i]],
                                                        system_prompts = [system_prompts[i]],
                                                        json_format = json_format,
                                                        temperature = temperature+0.5 if temperature<1 else temperature, 
                                                        images = None,
                                                        model=model,
                                                        options_generation = options_generation)
            if type(cleaned)==str:
                answer = json.loads(cleaned)
                answer = json_format(**answer)
            else:
                answer = cleaned
            results.append(answer)
        return results

    def predict_json(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        json_format: BaseModel,
        temperature=0,
        images: list[str] = None,
        options_generation=None,
    ) -> BaseModel:
        """
        It formats the queries with good prompts, then gives these prompts to the LLM and return the cleaned outputs following the given BaseModel format
        """
        return self.multiple_predict_json(prompts=[prompt],
                                          system_prompts=[system_prompt],
                                          model = model,
                                          json_format=json_format,
                                          temperature=temperature,
                                          images=[images],
                                          options_generation=options_generation)[0]

    def predict_image(
        self, prompt: str, model: str, 
        data_url, json_format: BaseModel, 
        temperature=0, options_generation=None
    ) -> BaseModel:
        """
        It formats the queries with good prompts, then gives these prompts to the LLM and return the cleaned outputs following the given BaseModel format
        """
        answer = self.multiple_predict(
            prompts=[prompt],
            system_prompts=[prompt],
            model = model,
            temperature=temperature,
            images=[data_url],
            options_generation=options_generation,
        )
        answer["texts"] = answer["texts"][0]
        return answer
    
    def embeddings(self, texts, model):
        if type(texts) is type("str"):
            texts = [texts]

        data = {"model_name": model, "texts": texts}
        url = self.url + "/embeddings"
        embeddings = requests.post(url, json=data)
        embeddings = embeddings.json()
        nb_tokens = embeddings["nb_tokens"]
        embeddings = embeddings["embeddings"]
        return {"embeddings": embeddings, "nb_tokens": nb_tokens}

    def embeddings_vlm(self, model, images=[], queries=[], mode="vlm"):
        if type(queries) == type("str"):
            queries = [queries]

        if type(images) == np.ndarray:
            images = [images]

        files = []
        for i in range(len(images)):
            files.append(
                (
                    "images",
                    ("0", np_array_to_file(image_np=np.array(images[i])), "image/jpeg"),
                )
            )

        data = {"model_name": model, "queries": queries, "mode": mode}
        url = self.url + "/embeddings_vlm"
        embeddings = requests.post(url, data=data, files=files)
        embeddings = embeddings.json()
        nb_tokens = embeddings["nb_tokens"]
        embeddings = np.array(embeddings["embeddings"])
        return {"embeddings": embeddings, "nb_tokens": nb_tokens}

    def reranking(self, query, chunk_list: list[Chunk], model):
        if type(chunk_list) is Chunk:
            chunk_list = [chunk_list]
        contexts = [chunk.text for chunk in chunk_list]
        data = {"model_name": model, "query": query, "contexts": contexts}
        url = self.url + "/reranking"
        scores = requests.post(url, json=data).json()
        return scores

    def release_memory(self):
        data = {}
        url = self.url + "/release_memory"
        scores = requests.post(url, json=data).json()



class Agent_openai(Agent):

    def __init__(self,
                 models_infos,
                 url: str = None,
                 language="EN",
                 max_attempts=5,
                 max_workers = 10):
        """
        Args :
            model : Name of the LLM used to realize the task (available LLMs are the keys of the dictionary from resources/prompts.json)
            key_or_url : Docker's port if ollama, api_key else if open ai
            language : Language of prompts
            max_attempts : Maximal number of attempts the LLM will try to give you a correct format for the answer
        """
        self.models_infos = models_infos
        self.clients = {}
        for key in self.models_infos.keys():
            if "api_key" in self.models_infos[key].keys() and "url" in self.models_infos[key].keys():
                if self.models_infos[key]["url"] is not None:
                    url = self.models_infos[key]["url"] + "/v1"
                self.clients[key] = OpenAI(api_key=self.models_infos[key]["api_key"],
                                           base_url=url)

        
        self.language = language
        self.max_attempts = max_attempts
        self.temperature = 0
        self.max_workers = max_workers

    def predict_image(self, 
                      prompt: str, 
                      model: str,
                      data_url,
                      json_format: BaseModel,
                      temperature=0) -> BaseModel:
        """
        It formats the queries with good prompts, then gives these prompts to the LLM and return the cleaned outputs following the given BaseModel format
        """
        answer = predict_image(
            prompt,
            model,
            data_url,
            self.clients[model],
            json_format,
            temperature,
        )
        return answer

    def predict(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float = 0,
        options_generation=None,
    ) -> str:
        """
        It formats the query with the good prompt, then gives this prompt to the LLM and return the cleaned output
        """
        temperature = self.temperature
        answer = predict(
            system_prompt,
            prompt,
            model,
            self.clients[model],
            temperature=temperature,
            options_generation=options_generation,
        )
        return answer
    

    def embeddings(self, texts, model):
        embeddings = self.clients[model].embeddings.create(input=texts,
                                                           model=model)

        if type(texts) is type([]):
            vector_embeddings = [
                embeddings.data[k].embedding for k in range(len(texts))
            ]

        else:
            vector_embeddings = [embeddings.data[0].embedding]

        return {
            "embeddings": vector_embeddings,
            "nb_tokens": embeddings.usage.total_tokens,
        }

    def reranking(self, query: str, chunk_list: list[Chunk],
                  model, max_workers: int = 10):
        if self.models_infos[model]["type"]=="reranker":
            documents = [chunk.text for chunk in chunk_list]
            url = self.models_infos[model]['url'] + "/v1/rerank"
            payload = {
                "model": model,
                "query": query,
                "documents": documents
            }

            headers = {
                "accept": "application/json",
                "Content-Type": "application/json"
            }
            response = requests.post(url,
                                     json=payload,
                                     headers=headers).json()
            input_tokens = [response["usage"]["total_tokens"]]
            ordered_by_index = sorted(response["results"], key=lambda x: x["index"])

            scores = [item["relevance_score"] for item in ordered_by_index]
        elif self.models_infos[model]["type"]=="embedding":
            documents = [chunk.text for chunk in chunk_list]
            emb_chunks = self.clients[model].embeddings.create(input=documents,
                                                              model=model)
            chunks_tokens = emb_chunks.usage.total_tokens

            emb_chunk = [emb_chunks.data[k].embedding for k in range(len(emb_chunks.data))]
            emb_query = self.clients[model].embeddings.create(input=query, 
                                                              model=model)
            query_tokens = emb_query.usage.total_tokens
            emb_query = emb_query.data[0].embedding
            scores = [cosine_similarity(doc, emb_query) for doc in emb_chunk]
            input_tokens = [chunks_tokens+query_tokens]
        elif self.models_infos[model]["type"]=="llm":
            system_prompt = """You are a highly accurate reranking model. Given a user query and a retrieved document chunk, your job is to assign a numerical relevance score from 0 to 1, where:

                                1.0 means "perfectly relevant",
                                0.0 means "completely irrelevant".

                                Evaluate the document chunk solely based on its relevance to answering or supporting the query. Do not hallucinate or infer information not present in the chunk."""
            scores = []
            input_tokens = []

            if max_workers <= get_executor_threads():
                max_workers = 1

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for chunk in chunk_list:
                    prompt = f" Context : {chunk.text}\n Query : {query}"
                    future = executor.submit(rerank,
                                            system_prompt,
                                            prompt,
                                            model,
                                            self.clients[model],
                                            temperature=None)
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    json_response, nb_input_tokens = future.result()
                    scores.append(json_response.score)
                    input_tokens.append(nb_input_tokens)


        return {
            "scores": scores,
            "nb_input_tokens": input_tokens,
            }

    def release_memory(self):
        None
