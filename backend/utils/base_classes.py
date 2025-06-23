from abc import ABC, abstractmethod
from typing import Union
import nltk
from .agent_functions import predict, multiple_predict
from pydantic import BaseModel


class Splitter(ABC):

    @abstractmethod
    def split_text(self, text: str, chunk_size: int, **kwargs) -> list[str]:
        """
        Split a text into chunks of size chunk_size
        """
        pass

    def break_chunks(self, chunks, max_size_chunk=512):
        final_chunks = []
        for i in range(len(chunks)):
            if len(chunks[i]) < max_size_chunk:
                final_chunks.append(chunks[i])
            else:
                sentences = nltk.sent_tokenize(chunks[i])
                current_chunk = sentences[0]
                for j in range(1, len(sentences)):
                    if len(current_chunk) + len(sentences[j]) < max_size_chunk:
                        current_chunk += sentences[j]
                    else:
                        final_chunks.append(current_chunk)
                        current_chunk = sentences[j]
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
    def add_str_elements(
        self,
        collection_name: str,
        elements: list[str],
        metadata: dict = None,
        display_message: bool = True,
    ) -> None:
        pass

    @abstractmethod
    def k_search(
        self,
        queries: Union[str, list[str]],
        k: int,
        collection_name: str,
        output_fields: list[str] = ["text"],
        filters: dict = None,
    ):
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
    def __init__(self, model: str, language: str = "EN", max_attempts: int = 5) -> None:
        self.model = model
        self.language = language
        self.max_attempts = max_attempts
        self.temperature = None

    def predict(
        self, prompt: str, system_prompt: str, with_tokens: bool = False
    ) -> str:
        """
        It generates the answer of the model given the prompt and the system_prompt
        """

        return predict(system_prompt, prompt, self.model, with_tokens)

    def multiple_predict(
        self, prompts: list[str], system_prompts: list[str]
    ) -> list[str]:
        """
        It batches and generates the answer of the model given the prompts and the system_prompts
        """

        return multiple_predict(system_prompts, prompts, self.model)

    def predict_json(
        self,
        system_prompt: str,
        prompt: str,
        json_format: BaseModel,
    ) -> BaseModel:

        pass
