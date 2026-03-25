from typing import Dict, Any, Union
from core.interfaces.llm_provider import LLMProvider
from core.error_handler import ConfigurationError, LLMError
from .openai_compatible_provider import OpenAICompatibleProvider
from .openai_provider import OpenAIProvider
from .mistral_provider import MistralProvider
from .anthropic_provider import AnthropicProvider
from .azure_openai_provider import AzureOpenAIProvider
from .cohere_provider import CohereProvider
from .gemini_provider import GeminiProvider
from .openrouter_provider import OpenRouterProvider
from .deepseek_provider import DeepSeekProvider
from .kimi_provider import KimiProvider
from .glm_provider import GLMProvider
from .groq_provider import GroqProvider
from .composite_provider import CompositeProvider

class LLMProviderFactory:
    _provider_classes = {
        'openai': OpenAIProvider,
        'mistral': MistralProvider,
        'anthropic': AnthropicProvider,
        'azure': AzureOpenAIProvider,
        'cohere': CohereProvider,
        'gemini': GeminiProvider,
        'openrouter': OpenRouterProvider,
        'deepseek': DeepSeekProvider,
        'kimi': KimiProvider,
        'glm': GLMProvider,
        'groq': GroqProvider,
        'custom': OpenAICompatibleProvider,
        'default': OpenAICompatibleProvider
    }

    @classmethod
    def create_provider(cls, provider_type: str, models_infos: Dict[str, Any], **kwargs) -> LLMProvider:
        provider_type = provider_type.lower()
        if provider_type not in cls._provider_classes:
            raise ConfigurationError(f'Unknown LLM provider type: {provider_type}', config_key='provider_type', config_value=provider_type)
        provider_class = cls._provider_classes[provider_type]
        try:
            return provider_class(models_infos=models_infos, **kwargs)
        except Exception as e:
            raise LLMError(f'Failed to create {provider_type} provider', provider=provider_type, original_error=e)

    @classmethod
    def create_all_providers(cls, models_infos: Dict[str, Any], language: str='EN', max_attempts: int=5, max_workers: int=10) -> Dict[str, LLMProvider]:
        try:
            result = {}
            
            for (model_name, model_config) in models_infos.items():
                has_api_key = model_config.get('api_key') and model_config.get('api_key').strip()
                has_url = model_config.get('url') and model_config.get('url').strip()
                has_azure_endpoint = model_config.get('azure_endpoint') and model_config.get('azure_endpoint').strip()
                model_type = model_config.get('type', 'llm')
                provider_type = model_config.get('provider', 'openai').lower()
                
                if has_api_key or has_url or has_azure_endpoint or model_type in ['llm', 'embedding', 'reranker']:
                    provider_class = cls._provider_classes.get(provider_type, OpenAICompatibleProvider)
                    provider = provider_class(
                        models_infos={model_name: model_config},
                        language=language,
                        max_attempts=max_attempts,
                        max_workers=max_workers
                    )
                    result[model_name] = provider
            return result
        except Exception as e:
            raise LLMError('Failed to create LLM providers', original_error=e)

    @classmethod
    def register_provider(cls, provider_type: str, provider_class: type) -> None:
        if not issubclass(provider_class, LLMProvider):
            raise ConfigurationError(f'Provider class must implement LLMProvider interface', config_key='provider_class', config_value=provider_class.__name__)
        cls._provider_classes[provider_type.lower()] = provider_class

    @classmethod
    def create_composite_provider(cls, models_infos: Dict[str, Any], language: str='EN', max_attempts: int=5, max_workers: int=10) -> CompositeProvider:
        """
        Crée un provider composite qui agrège tous les providers et route
        automatiquement vers le bon provider selon le modèle demandé.
        """
        providers = cls.create_all_providers(models_infos, language, max_attempts, max_workers)
        return CompositeProvider(providers=providers)

    @classmethod
    def get_available_providers(cls) -> list[str]:
        return list(cls._provider_classes.keys())

def get_llm_provider(provider_type: str='default', models_infos: Dict[str, Any]=None, **kwargs) -> LLMProvider:
    if models_infos is None:
        raise ConfigurationError('models_infos is required for creating LLM provider', config_key='models_infos')
    return LLMProviderFactory.create_provider(provider_type=provider_type, models_infos=models_infos, **kwargs)
