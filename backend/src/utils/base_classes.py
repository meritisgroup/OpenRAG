from abc import ABC, abstractmethod
from typing import Union
import nltk
from .agent_functions import predict, multiple_predict, predict_json
from pydantic import BaseModel
from typing import List
from .threading_utils import get_executor_threads
import concurrent.futures

class Splitter(ABC):

    @abstractmethod
    def split_text(self, text: str, chunk_size: int, **kwargs) -> list[str]:
        pass

    def break_chunks(self, chunks, max_size_chunk=512):
        final_chunks = []
        for text in chunks:
            if len(text) < max_size_chunk:
                final_chunks.append(text)
            else:
                words = text.split()
                if words:
                    current_chunk = words[0]
                    for w in words[1:]:
                        if len(current_chunk) + 1 + len(w) <= max_size_chunk:
                            current_chunk += ' ' + w
                        else:
                            final_chunks.append(current_chunk)
                            current_chunk = w
                    final_chunks.append(current_chunk)
                else:
                    print(f"CE TEXTE N'EST PAS PASSE {text}")
        return final_chunks

class VectorBase:

    @abstractmethod
    def __init__(self, vb_name: str, path: str) -> None:
        self.vb_name = vb_name
        self.nb_tokens_embeddings: int = 0
        pass

    @abstractmethod
    def create_collection(self, name: str) -> None:
        pass

    @abstractmethod
    def add_str_elements(self, collection_name: str, elements: list[str], metadata: dict=None, display_message: bool=True) -> None:
        pass

    @abstractmethod
    def k_search(self, queries: Union[str, list[str]], k: int, collection_name: str, output_fields: list[str]=['text'], filters: dict=None):
        pass

    def get_collection_name(self):
        return self.vb_name

    @abstractmethod
    def add_name_done_doc(self):
        pass

    def get_nb_token_embeddings(self):
        return self.nb_tokens_embeddings

class Agent:

    @abstractmethod
    def __init__(self, language: str='EN', max_attempts: int=5) -> None:
        self.language = language
        self.max_attempts = max_attempts
        self.temperature = None

    def predict(self, prompt: str, model: str, system_prompt: str, with_tokens: bool=False) -> str:
        return predict(system_prompt, prompt, model, with_tokens)

    def multiple_predict_json(self, prompts: list[str], system_prompts: list[str], model: str, json_format: BaseModel, temperature=0, images: list[list[str]]=None, options_generation=None, max_workers: int=10):
        results = [None] * len(prompts)
        if images is None:
            images = [None] * len(prompts)
        if max_workers <= get_executor_threads():
            max_workers = 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self.predict_json, prompt=prompts[i], system_prompt=system_prompts[i], model=model, json_format=json_format, temperature=temperature, options_generation=options_generation): i for i in range(len(prompts))}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    print(f'Prompt #{index} a généré une exception : {exc}')
        return results

    def multiple_predict(self, model: str, prompts: List[str], system_prompts: List[str], temperature: float=0, options_generation=None) -> str:
        answers = multiple_predict(system_prompts, prompts, model, self.clients[model], temperature=temperature, options_generation=options_generation, max_workers=self.max_workers)
        return answers

    def predict_json(self, prompt: str, model: str, system_prompt: str, json_format: BaseModel, temperature=0, options_generation=None) -> BaseModel:
        answer = predict_json(system_prompt, prompt, model, self.clients[model], json_format, temperature, options_generation=options_generation)
        return answer