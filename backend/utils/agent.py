from .agent_functions import (
    predict,
    multiple_predict,
    predict_json,
    predict_images,
    predict_image,
    predict_mistral,
    multiple_predict_mistral,
    rerank,
    RerankedChunk,
)
from ecologits import EcoLogits
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
                else:
                    url = None

                self.clients[key] = OpenAI(api_key=self.models_infos[key]["api_key"],
                                           base_url=url)

        
        self.language = language
        self.max_attempts = max_attempts
        self.temperature = 0
        self.max_workers = max_workers

    def predict_images(self, 
                      prompts: list[str],
                      model: str,
                      images: list[np.ndarray],
                      json_format: BaseModel = None,
                      temperature: float = None,
                      max_workers: int = 10):
        return predict_images(prompts=prompts,
                                model=model,
                                images=images,
                                client=self.clients[model],
                                json_format=json_format,
                                temperature=temperature,
                                max_workers=max_workers)
    
    def predict_image(self, 
                      prompt: str,
                      model: str,
                      image: np.ndarray,
                      json_format: BaseModel = None,
                      temperature: float = None):
        return predict_image(prompt=prompt,
                             model=model,
                             img=image,
                             client=self.clients[model],
                             json_format=json_format,
                             temperature=temperature)
        

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
        if not getattr(EcoLogits, "_initialized", False):
            EcoLogits.init()
            EcoLogits._initialized = True

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
            a = time.time()
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
                future_to_index = {}
                for i, chunk in enumerate(chunk_list):
                    prompt = f" Context : {chunk.text}\n Query : {query}"
                    future = executor.submit(rerank,
                                            system_prompt,
                                            prompt,
                                            model,
                                            self.clients[model],
                                            temperature=None)
                    future_to_index[future] = i 

                unordered_results = []
                for future in concurrent.futures.as_completed(future_to_index):
                    original_index = future_to_index[future]
                    try:
                        json_response, nb_input_tokens = future.result()
                        unordered_results.append((original_index, json_response.score, nb_input_tokens))
                    except Exception as exc:
                        print(f'La tâche {original_index} a échoué: {exc}')

                unordered_results.sort(key=lambda x: x[0])

                scores = [result[1] for result in unordered_results]
                input_tokens = [result[2] for result in unordered_results]

        return {"scores": scores,
                "nb_input_tokens": input_tokens}

    def release_memory(self):
        None
