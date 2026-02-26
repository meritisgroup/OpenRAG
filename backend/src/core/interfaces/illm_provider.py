from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel
import numpy as np

class ILLMProvider(ABC):

    @abstractmethod
    def predict(self, prompt: str, system_prompt: str, model: str, temperature: float=0, options_generation: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def multiple_predict(self, prompts: List[str], system_prompt: str, model: str, temperature: float=0, max_workers: int=10) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict_json(self, prompt: str, system_prompt: str, model: str, json_format: type[BaseModel], temperature: float=0) -> Optional[BaseModel]:
        pass

    @abstractmethod
    def embeddings(self, texts: Union[str, List[str]], model: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict_image(self, prompt: str, model: str, image: np.ndarray, json_format: Optional[type[BaseModel]]=None, temperature: float=0) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict_images(self, prompts: List[str], model: str, images: List[np.ndarray], json_format: Optional[type[BaseModel]]=None, temperature: float=0, max_workers: int=10) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def reranking(self, query: str, chunk_list: List, model: str, max_workers: int=10) -> Dict[str, Any]:
        pass

    @abstractmethod
    def release_memory(self) -> None:
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        pass