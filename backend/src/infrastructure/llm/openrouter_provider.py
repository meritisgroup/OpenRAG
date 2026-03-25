from typing import Dict, Any
from .openai_compatible_provider import OpenAICompatibleProvider

OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'

class OpenRouterProvider(OpenAICompatibleProvider):

    def __init__(self, models_infos: Dict[str, Any], language: str='EN', max_attempts: int=5, max_workers: int=10):
        for key in models_infos:
            if 'url' not in models_infos[key] or not models_infos[key]['url']:
                models_infos[key]['url'] = OPENROUTER_BASE_URL
        super().__init__(models_infos, language, max_attempts, max_workers)

    @property
    def provider_name(self) -> str:
        return 'openrouter'
