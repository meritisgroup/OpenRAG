from typing import Dict, Any, Union
from core.interfaces.illm_provider import ILLMProvider
from core.error_handler import ConfigurationError, LLMError
from .base_provider import BaseLLMProvider
from .openai_provider import OpenAIProvider
from .mistral_provider import MistralProvider
from .ollama_provider import OllamaProvider
from .vllm_provider import VLLMProvider

class LLMProviderFactory:
    _provider_classes = {'openai': OpenAIProvider, 'mistral': MistralProvider, 'ollama': OllamaProvider, 'vllm': VLLMProvider, 'default': BaseLLMProvider}

    @classmethod
    def create_provider(cls, provider_type: str, models_infos: Dict[str, Any], **kwargs) -> ILLMProvider:
        provider_type = provider_type.lower()
        if provider_type not in cls._provider_classes:
            raise ConfigurationError(f'Unknown LLM provider type: {provider_type}', config_key='provider_type', config_value=provider_type)
        provider_class = cls._provider_classes[provider_type]
        try:
            return provider_class(models_infos=models_infos, **kwargs)
        except Exception as e:
            raise LLMError(f'Failed to create {provider_type} provider', provider=provider_type, original_error=e)

    @classmethod
    def create_all_providers(cls, models_infos: Dict[str, Any], language: str='EN', max_attempts: int=5, max_workers: int=10) -> Dict[str, ILLMProvider]:
        try:
            provider = BaseLLMProvider(models_infos=models_infos, language=language, max_attempts=max_attempts, max_workers=max_workers)
            result = {}
            for (model_name, model_config) in models_infos.items():
                has_api_key = model_config.get('api_key') and model_config.get('api_key').strip()
                has_url = model_config.get('url') and model_config.get('url').strip()
                model_type = model_config.get('type', 'llm')
                if has_api_key or has_url or model_type in ['llm', 'embedding', 'reranker']:
                    result[model_name] = provider
            return result
        except Exception as e:
            raise LLMError('Failed to create LLM providers', original_error=e)

    @classmethod
    def register_provider(cls, provider_type: str, provider_class: type) -> None:
        if not issubclass(provider_class, ILLMProvider):
            raise ConfigurationError(f'Provider class must implement ILLMProvider interface', config_key='provider_class', config_value=provider_class.__name__)
        cls._provider_classes[provider_type.lower()] = provider_class

    @classmethod
    def get_available_providers(cls) -> list[str]:
        return list(cls._provider_classes.keys())

def get_llm_provider(provider_type: str='default', models_infos: Dict[str, Any]=None, **kwargs) -> ILLMProvider:
    if models_infos is None:
        raise ConfigurationError('models_infos is required for creating LLM provider', config_key='models_infos')
    return LLMProviderFactory.create_provider(provider_type=provider_type, models_infos=models_infos, **kwargs)