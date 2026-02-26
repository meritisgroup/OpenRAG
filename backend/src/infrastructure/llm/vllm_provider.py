from typing import Dict, Any
from .base_provider import BaseLLMProvider
from core.error_handler import LLMError

class VLLMProvider(BaseLLMProvider):

    def __init__(self, models_infos: Dict[str, Any], language: str='EN', max_attempts: int=5, max_workers: int=10, base_url: str=None):
        super().__init__(models_infos, language, max_attempts, max_workers)
        self.base_url = base_url

    @property
    def provider_name(self) -> str:
        return 'vllm'