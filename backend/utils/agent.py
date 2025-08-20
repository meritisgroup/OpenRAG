from .agent_functions import (
    predict,
    multiple_predict,
    predict_vllm,
    predict_json,
    predict_mistral,
    multiple_predict_mistral,
    rerank,
    RerankedChunk,
)
from openai import OpenAI
from mistralai import Mistral
import json
import requests
import time
import numpy as np
from typing import List
from .base_classes import Agent
from .agent_functions import np_array_to_file
from pydantic import BaseModel


def get_Agent(
    config_server: dict,
):
    params_host_llm = config_server["params_host_llm"]
    if params_host_llm["type"] == "ollama":

        agent = Agent_ollama(
            model=config_server["model"],
            key_or_url=params_host_llm["url"],
            language=config_server["language"],
            max_attempts=config_server["max_attempts"],
        )
    if params_host_llm["type"] == "vllm":
        agent = Agent_Vllm(
            model=config_server["model"],
            key_or_url=params_host_llm["url"],
            language=config_server["language"],
            max_attempts=config_server["max_attempts"],
        )
    elif params_host_llm["type"] == "openai":
        agent = Agent_openai(
            model=config_server["model"],
            key_or_url=params_host_llm["api_key"],
            language=config_server["language"],
            max_attempts=config_server["max_attempts"],
        )
    elif params_host_llm["type"] == "mistral":
        agent = Agent_mistral(
            model=config_server["model"],
            key_or_url=params_host_llm["api_key"],
            language=config_server["language"],
            max_attempts=config_server["max_attempts"],
        )
    elif params_host_llm["type"] == "mistral":
        agent = Agent_mistral(
            model=config_server["model"],
            key_or_url=params_host_llm["api_key"],
            language=config_server["language"],
            max_attempts=config_server["max_attempts"],
        )
    return agent


class Agent_ollama(Agent):

    def __init__(
        self,
        model: str,
        key_or_url="http://localhost:11434/v1",
        language="EN",
        max_attempts=5,
    ):
        """
        Args :
            model : Name of the LLM used to realize the task (available LLMs are the keys of the dictionary from resources/prompts.json)
            key_or_url : Docker's port if ollama, api_key else if open ai
            language : Language of prompts
            max_attempts : Maximal number of attempts the LLM will try to give you a correct format for the answer
        """
        self.client = OpenAI(base_url=key_or_url, api_key="ollama")

        self.model = model
        self.language = language
        self.max_attempts = max_attempts
        self.temperature = 0

    def predict(self, prompt: str, system_prompt: str, temperature=0, options_generation = None) -> str:
        """
        It formats the query with the good prompt, then gives this prompt to the LLM and return the cleaned output
        """
        answer = predict(system_prompt, prompt, self.model, self.client, temperature, options_generation=options_generation)
        return answer

    def predict_json(
        self, prompt: str, system_prompt: str, json_format: BaseModel, temperature=0,
        options_generation = None
    ) -> BaseModel:
        """
        It formats the queries with good prompts, then gives these prompts to the LLM and return the cleaned outputs following the given BaseModel format
        """
        answer = predict_json(
            system_prompt,
            prompt,
            self.model,
            self.client,
            json_format,
            temperature,
            options_generation=options_generation
        )
        return answer

    def multiple_predict(
        self, prompts: List[str], system_prompts: List[str], temperature=0, 
        options_generation = None
    ) -> str:
        """
        It formats the queries with good prompts, then gives these prompts to the LLM and return the cleaned outputs
        """
        answers = multiple_predict(
            system_prompts, prompts, self.model, self.client, temperature, options_generation=options_generation
        )
        return answers

    def embeddings(self, texts, model):
        embeddings = self.client.embeddings.create(input=texts, model=model)

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

    def reranking(self, query: str, contexts: list[str], model: str = "gemma3:1b"):
        scores = []
        input_tokens = []
        system_prompt = """You are a highly accurate reranking model. Given a user query and a retrieved document chunk, your job is to assign a numerical relevance score from 0 to 1, where:

                            1.0 means "perfectly relevant",
                            0.0 means "completely irrelevant".

                            Evaluate the document chunk solely based on its relevance to answering or supporting the query. Do not hallucinate or infer information not present in the chunk."""
        for context in contexts:
            prompt = f" Context : {context}\n Query : {query}"
            json_response, nb_input_tokens = rerank(
                system_prompt, prompt, model, self.client, temperature=None
            )

            scores.append(json_response.score)
            input_tokens.append(nb_input_tokens)

        return {
            "scores": scores,
            "nb_input_tokens": input_tokens,
        }

    def release_memory(self):
        None


class Agent_Vllm(Agent):
    def __init__(
        self,
        model: str,
        key_or_url="http://0.0.0.0:8000",
        language="EN",
        max_attempts=5,
    ):
        self.model = model
        self.language = language
        self.max_attempts = max_attempts
        self.url = key_or_url
        self.temperature = 0

    def multiple_predict(
        self,
        prompts: List[str],
        system_prompts: List[str],
        images: list[list[str]] = None,
        json_format=None,
        temperature=0,
        options_generation = None
    ) -> str:
        """
        It formats the query with the good prompt, then gives this prompt to the LLM and return the cleaned output
        """

        url = self.url + "/predict"
        answers = predict_vllm(
            system_prompts=system_prompts,
            prompts=prompts,
            model=self.model,
            url=url,
            temperature=self.temperature,
            images=images,
            json_format=json_format,
            options_generation=options_generation
        )

        return answers

    def predict_json(
        self,
        prompt: str,
        system_prompt: str,
        json_format: BaseModel,
        temperature=0,
        images: list[str] = None,
        options_generation = None
    ) -> BaseModel:
        """
        It formats the queries with good prompts, then gives these prompts to the LLM and return the cleaned outputs following the given BaseModel format
        """
        answer = self.multiple_predict(prompts=[prompt],
                                       system_prompts=[system_prompt],
                                       temperature=temperature,
                                       json_format=json_format,
                                       images=[images],
                                       options_generation=options_generation)
        answer = answer["texts"][0]

        if "json" in answer:
            cleaned = answer.split("json\n")[1].split("\n```")[0]
        elif "{" in answer:
            cleaned = answer.split("{")[1].split("}")[0]
            cleaned = "{" + cleaned + "}"
        elif "True" in answer and json_format.__name__ == "StatementSupported":
            cleaned = '{"supported": true}'

        elif "False" in answer and json_format.__name__ == "StatementSupported":
            cleaned = '{"supported": false}'
        else:
            """
            json_format_str = json.dumps(json_format.model_json_schema())
            prompt = "Can you rewrite the answer to make it match the given json format: \n\n Here's the answer:{}\n\n Can you rewrite it to match this JSON format: {}".format(answer, str(json_format_str))
            system_prompt = "You are an AI assistant which has for mission to rewrite the answer to match a given json format"
            return self.predict_json(prompt=prompt,
                                     system_prompt=system_prompt,
                                     json_format=json_format,
                                     temperature=temperature,
                                     images=images)
            """
        answer = json.loads(cleaned)
        answer = json_format(**answer)
        return answer

    def predict(
        self, prompt: str, system_prompt: str, images: list[str] = None, temperature=0, options_generation = None
    ) -> str:
        answer = self.multiple_predict(
            prompts=[prompt],
            system_prompts=[system_prompt],
            temperature=temperature,
            images=[images],
            options_generation = options_generation
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

    def reranking(self, query, contexts, model):
        if type(contexts) is str:
            contexts = [contexts]
        data = {"model_name": model, "query": query, "contexts": contexts}
        url = self.url + "/reranking"
        scores = requests.post(url, json=data).json()
        return scores

    def release_memory(self):
        data = {}
        url = self.url + "/release_memory"
        scores = requests.post(url, json=data).json()


class Agent_openai(Agent):

    def __init__(
        self,
        model: str,
        key_or_url: str = None,
        language="EN",
        max_attempts=5,
    ):
        """
        Args :
            model : Name of the LLM used to realize the task (available LLMs are the keys of the dictionary from resources/prompts.json)
            key_or_url : Docker's port if ollama, api_key else if open ai
            language : Language of prompts
            max_attempts : Maximal number of attempts the LLM will try to give you a correct format for the answer
        """
        self.client = OpenAI(api_key=key_or_url)

        self.model = model
        self.language = language
        self.max_attempts = max_attempts
        self.temperature = 0

    def predict_json(
        self, prompt: str, system_prompt: str, json_format: BaseModel, temperature=0, options_generation = None
    ) -> BaseModel:
        """
        It formats the queries with good prompts, then gives these prompts to the LLM and return the cleaned outputs following the given BaseModel format
        """
        temperature = self.temperature
        answer = predict_json(
            system_prompt,
            prompt,
            self.model,
            self.client,
            json_format,
            temperature,
            options_generation=options_generation
        )
        return answer

    def predict(self, prompt: str, system_prompt: str, temperature: float = 0, 
        options_generation = None) -> str:
        """
        It formats the query with the good prompt, then gives this prompt to the LLM and return the cleaned output
        """
        temperature = self.temperature
        answer = predict(
            system_prompt, prompt, self.model, self.client, temperature=temperature, options_generation=options_generation
        )
        return answer

    def multiple_predict(
        self, prompts: List[str], system_prompts: List[str], temperature: float = 0, 
        options_generation = None
    ) -> str:
        """
        It formats the queries with good prompts, then gives these prompts to the LLM and return the cleaned outputs
        """
        temperature = self.temperature
        answers = multiple_predict(
            system_prompts, prompts, self.model, self.client, temperature=temperature, options_generation=options_generation
        )
        return answers

    def embeddings(self, texts, model):
        embeddings = self.client.embeddings.create(input=texts, model=model)

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

    def reranking(self, query: str, contexts: list[str], model: str = "gemma3:1b"):
        scores = []
        input_tokens = []
        system_prompt = """You are a highly accurate reranking model. Given a user query and a retrieved document chunk, your job is to assign a numerical relevance score from 0 to 1, where:

                            1.0 means "perfectly relevant",
                            0.0 means "completely irrelevant".

                            Evaluate the document chunk solely based on its relevance to answering or supporting the query. Do not hallucinate or infer information not present in the chunk."""

        for context in contexts:
            prompt = f" Context : {context}\n Query : {query}"
            json_response, nb_input_tokens = rerank(
                system_prompt, prompt, model, self.client, temperature=None
            )

            scores.append(json_response.score)
            input_tokens.append(nb_input_tokens)

        return {
            "scores": scores,
            "nb_input_tokens": input_tokens,
        }

    def release_memory(self):
        None


class Agent_mistral(Agent_openai):

    def __init__(
        self,
        model: str,
        key_or_url: str = None,
        language="EN",
        max_attempts=5,
    ):
        """
        Args :
            model : Name of the LLM used to realize the task (available LLMs are the keys of the dictionary from resources/prompts.json)
            key_or_url : Docker's port if ollama, api_key else if open ai
            language : Language of prompts
            max_attempts : Maximal number of attempts the LLM will try to give you a correct format for the answer
        """
        super().__init__(
            model=model,
            key_or_url=key_or_url,
            language=language,
            max_attempts=max_attempts,
        )

        self.client = Mistral(api_key=key_or_url)

        self.model = model
        self.language = language
        self.max_attempts = max_attempts

    def embeddings(self, texts, model):
        embeddings = self.client.embeddings.create(inputs=texts, model=model)

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

    def predict_json(
        self, prompt: str,
        system_prompt: str,
        json_format: BaseModel,
        temperature=0,
        options_generation = None
    ) -> BaseModel:
            """
            It formats the queries with good prompts, then gives these prompts to the LLM and return the cleaned outputs following the given BaseModel format
            """
            temperature=self.temperature
            answer = predict_json(
                system_prompt,
                prompt,
                self.model,
                self.client,
                json_format,
                temperature,
                options_generation=options_generation
            )
            return answer
    
    def predict(self, prompt: str, system_prompt: str, temperature: float = 0, options_generation = None) -> str:
        """
        It formats the query with the good prompt, then gives this prompt to the LLM and return the cleaned output
        """
        answer = predict_mistral(
            system_prompt, prompt, self.model, self.client, temperature=temperature, options_generation=options_generation
        )

        return answer

    def multiple_predict(
        self, prompts: List[str], system_prompts: List[str], temperature: float = 0, 
        options_generation = None
    ) -> str:
        """
        It formats the queries with good prompts, then gives these prompts to the LLM and return the cleaned outputs
        """
        answers = multiple_predict_mistral(
            system_prompts, prompts, self.model, self.client, temperature=temperature, options_generation=options_generation
        )
        return answers

    def reranking(self, query: str, contexts: list[str], model: str = "gemma3:1b"):
        scores = []
        input_tokens = []
        system_prompt = """You are a highly accurate reranking model. Given a user query and a retrieved document chunk, your job is to assign a numerical relevance score from 0 to 1, where:

                            1.0 means "perfectly relevant",
                            0.0 means "completely irrelevant".

                            Evaluate the document chunk solely based on its relevance to answering or supporting the query. Do not hallucinate or infer information not present in the chunk."""
        for context in contexts:
            prompt = f" Context : {context}\n Query : {query}"
            json_response, nb_input_tokens = rerank(
                system_prompt, prompt, model, self.client, temperature=None
            )

            scores.append(json_response.score)
            input_tokens.append(nb_input_tokens)

        return {
            "scores": scores,
            "nb_input_tokens": input_tokens,
        }
