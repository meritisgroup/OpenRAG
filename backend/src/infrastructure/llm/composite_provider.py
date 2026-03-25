from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel
import numpy as np
from core.interfaces.llm_provider import LLMProvider
from core.error_handler import LLMError

class CompositeProvider(LLMProvider):
    """
    Provider composite qui agrège tous les providers et route
    automatiquement vers le bon provider selon le modèle demandé.
    """

    def __init__(self, providers: Dict[str, LLMProvider]):
        self.providers = providers

    def _get_provider(self, model: str) -> LLMProvider:
        """Route vers le bon provider selon le modèle demandé."""
        if model in self.providers:
            return self.providers[model]
        available_models = list(self.providers.keys())
        raise LLMError(
            f"Model '{model}' not found in available providers",
            model=model,
            available_models=available_models[:10],
            provider='CompositeProvider'
        )

    def predict(self, prompt: str, system_prompt: str, model: str, temperature: float=0, options_generation: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        provider = self._get_provider(model)
        return provider.predict(prompt, system_prompt, model, temperature, options_generation)

    def multiple_predict(self, prompts: List[str], system_prompt: str, model: str, temperature: float=0, max_workers: int=10) -> Dict[str, Any]:
        provider = self._get_provider(model)
        return provider.multiple_predict(prompts, system_prompt, model, temperature, max_workers)

    def predict_json(self, prompt: str, system_prompt: str, model: str, json_format: type[BaseModel], temperature: float=0) -> Optional[BaseModel]:
        provider = self._get_provider(model)
        return provider.predict_json(prompt, system_prompt, model, json_format, temperature)

    def embeddings(self, texts: Union[str, List[str]], model: str, input_type: Optional[str] = None) -> Dict[str, Any]:
        provider = self._get_provider(model)
        return provider.embeddings(texts, model, input_type)

    def predict_image(self, prompt: str, model: str, image: np.ndarray, json_format: Optional[type[BaseModel]]=None, temperature: float=0) -> Dict[str, Any]:
        provider = self._get_provider(model)
        return provider.predict_image(prompt, model, image, json_format, temperature)

    def predict_images(self, prompts: List[str], model: str, images: List[np.ndarray], json_format: Optional[type[BaseModel]]=None, temperature: float=0, max_workers: int=10) -> List[Dict[str, Any]]:
        provider = self._get_provider(model)
        return provider.predict_images(prompts, model, images, json_format, temperature, max_workers)

    def reranking(self, query: str, chunk_list: List, model: str, max_workers: int=10) -> Dict[str, Any]:
        provider = self._get_provider(model)
        return provider.reranking(query, chunk_list, model, max_workers)

    def release_memory(self) -> None:
        for provider in self.providers.values():
            provider.release_memory()

    @property
    def is_available(self) -> bool:
        return len(self.providers) > 0

    @property
    def provider_name(self) -> str:
        return 'composite'
