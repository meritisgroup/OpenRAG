import concurrent.futures
import time
from typing import Dict, List, Any, Optional, Union
import requests
from pydantic import BaseModel
import numpy as np
from openai import OpenAI
from utils.ecologits_init import init_ecologits
from core.interfaces.illm_provider import ILLMProvider
from core.error_handler import LLMError
from utils.agent_functions import predict, multiple_predict, predict_json, predict_images, predict_image, rerank, RerankedChunk
from utils.threading_utils import get_executor_threads

class BaseLLMProvider(ILLMProvider):

    def __init__(self, models_infos: Dict[str, Any], language: str='EN', max_attempts: int=5, max_workers: int=10):
        self.models_infos = models_infos
        self.language = language
        self.max_attempts = max_attempts
        self.max_workers = max_workers
        self.temperature = 0.0
        init_ecologits()
        self.clients = self._create_clients()

    def _create_clients(self) -> Dict[str, OpenAI]:
        clients = {}
        for key in self.models_infos.keys():
            if 'api_key' in self.models_infos[key] and 'url' in self.models_infos[key]:
                url = self.models_infos[key]['url']
                if url is not None:
                    url = url + '/v1' if not url.endswith('/v1') else url
                clients[key] = OpenAI(api_key=self.models_infos[key]['api_key'], base_url=url)
        return clients

    def predict(self, prompt: str, system_prompt: str, model: str, temperature: float=0, options_generation: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        try:
            temperature = temperature or self.temperature
            answer = predict(system_prompt, prompt, model, self.clients[model], temperature=temperature, options_generation=options_generation)
            return answer
        except Exception as e:
            raise LLMError(f'Prediction failed for model {model}', provider=self.__class__.__name__, model=model, original_error=e)

    def multiple_predict(self, prompts: List[str], system_prompt: str, model: str, temperature: float=0, max_workers: int=10) -> Dict[str, Any]:
        try:
            answers = multiple_predict([system_prompt] * len(prompts), prompts, model, self.clients[model], temperature=temperature, max_workers=max_workers)
            return answers
        except Exception as e:
            raise LLMError(f'Multiple predictions failed for model {model}', provider=self.__class__.__name__, model=model, original_error=e)

    def predict_json(self, prompt: str, system_prompt: str, model: str, json_format: type[BaseModel], temperature: float=0) -> Optional[BaseModel]:
        try:
            response = predict_json(system_prompt, prompt, model, self.clients[model], json_format, temperature=temperature)
            return response
        except Exception as e:
            raise LLMError(f'JSON prediction failed for model {model}', provider=self.__class__.__name__, model=model, original_error=e)

    def embeddings(self, texts: Union[str, List[str]], model: str) -> Dict[str, Any]:
        try:
            embeddings = self.clients[model].embeddings.create(input=texts, model=model)
            if isinstance(texts, list):
                vector_embeddings = [embeddings.data[k].embedding for k in range(len(texts))]
            else:
                vector_embeddings = [embeddings.data[0].embedding]
            return {'embeddings': vector_embeddings, 'nb_tokens': embeddings.usage.total_tokens}
        except Exception as e:
            raise LLMError(f'Embedding generation failed for model {model}', provider=self.__class__.__name__, model=model, original_error=e)

    def predict_image(self, prompt: str, model: str, image: np.ndarray, json_format: Optional[type[BaseModel]]=None, temperature: float=0) -> Dict[str, Any]:
        try:
            return predict_image(prompt=prompt, model=model, img=image, client=self.clients[model], json_format=json_format, temperature=temperature)
        except Exception as e:
            raise LLMError(f'Image prediction failed for model {model}', provider=self.__class__.__name__, model=model, original_error=e)

    def predict_images(self, prompts: List[str], model: str, images: List[np.ndarray], json_format: Optional[type[BaseModel]]=None, temperature: float=0, max_workers: int=10) -> List[Dict[str, Any]]:
        try:
            return predict_images(prompts=prompts, model=model, images=images, client=self.clients[model], json_format=json_format, temperature=temperature, max_workers=max_workers)
        except Exception as e:
            raise LLMError(f'Multiple image predictions failed for model {model}', provider=self.__class__.__name__, model=model, original_error=e)

    def reranking(self, query: str, chunk_list: List, model: str, max_workers: int=10) -> Dict[str, Any]:
        try:
            model_type = self.models_infos[model].get('type', 'llm')
            if model_type == 'reranker':
                documents = [chunk.text for chunk in chunk_list]
                url = self.models_infos[model]['url'] + '/v1/rerank'
                payload = {'model': model, 'query': query, 'documents': documents}
                headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
                response = requests.post(url, json=payload, headers=headers).json()
                input_tokens = [response['usage']['total_tokens']]
                ordered_by_index = sorted(response['results'], key=lambda x: x['index'])
                scores = [item['relevance_score'] for item in ordered_by_index]
            elif model_type == 'embedding':
                documents = [chunk.text for chunk in chunk_list]
                emb_chunks = self.clients[model].embeddings.create(input=documents, model=model)
                chunks_tokens = emb_chunks.usage.total_tokens
                emb_chunk = [emb_chunks.data[k].embedding for k in range(len(emb_chunks.data))]
                emb_query = self.clients[model].embeddings.create(input=query, model=model)
                query_tokens = emb_query.usage.total_tokens
                emb_query = emb_query.data[0].embedding

                def cosine_similarity(a, b):
                    (a, b) = (np.array(a), np.array(b))
                    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                scores = [cosine_similarity(doc, emb_query) for doc in emb_chunk]
                input_tokens = [chunks_tokens + query_tokens]
            else:
                system_prompt = 'You are a highly accurate reranking model. Given a user query and a retrieved document chunk, your job is to assign a numerical relevance score from 0 to 1, where:\n                                1.0 means "perfectly relevant",\n                                0.0 means "completely irrelevant".\n                                Evaluate the document chunk solely based on its relevance to answering or supporting the query. Do not hallucinate or infer information not present in the chunk.'
                scores = []
                input_tokens = []
                if max_workers <= get_executor_threads():
                    max_workers = 1
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {}
                    for (i, chunk) in enumerate(chunk_list):
                        prompt = f' Context : {chunk.text}\n Query : {query}'
                        future = executor.submit(rerank, system_prompt, prompt, model, self.clients[model], temperature=0.0)
                        future_to_index[future] = i
                    unordered_results = []
                    for future in concurrent.futures.as_completed(future_to_index):
                        original_index = future_to_index[future]
                        try:
                            (json_response, nb_input_tokens) = future.result()
                            unordered_results.append((original_index, json_response.score, nb_input_tokens))
                        except Exception as exc:
                            print(f'Task {original_index} failed: {exc}')
                    unordered_results.sort(key=lambda x: x[0])
                    scores = [result[1] for result in unordered_results]
                    input_tokens = [result[2] for result in unordered_results]
            return {'scores': scores, 'nb_input_tokens': input_tokens}
        except Exception as e:
            raise LLMError(f'Reranking failed for model {model}', provider=self.__class__.__name__, model=model, original_error=e)

    def release_memory(self) -> None:
        pass

    @property
    def is_available(self) -> bool:
        return len(self.clients) > 0