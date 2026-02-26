from typing import Dict, Any
from .base_provider import BaseLLMProvider
from core.error_handler import LLMError

class MistralProvider(BaseLLMProvider):

    def __init__(self, models_infos: Dict[str, Any], language: str='EN', max_attempts: int=5, max_workers: int=10, api_key: str=None):
        super().__init__(models_infos, language, max_attempts, max_workers)
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return 'mistral'