from typing import Dict, Any
from openai import AzureOpenAI
from .openai_compatible_provider import OpenAICompatibleProvider
from core.error_handler import LLMError

class AzureOpenAIProvider(OpenAICompatibleProvider):

    def __init__(self, models_infos: Dict[str, Any], language: str='EN', max_attempts: int=5, max_workers: int=10,
                 azure_endpoint: str=None, api_version: str='2024-02-01'):
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        super().__init__(models_infos, language, max_attempts, max_workers)

    def _create_clients(self) -> Dict[str, AzureOpenAI]:
        clients = {}
        for deployment_name, config in self.models_infos.items():
            api_key = config.get('api_key')
            endpoint = config.get('azure_endpoint') or self.azure_endpoint
            api_ver = config.get('api_version') or self.api_version
            
            if api_key and endpoint:
                try:
                    clients[deployment_name] = AzureOpenAI(
                        api_key=api_key,
                        api_version=api_ver,
                        azure_endpoint=endpoint
                    )
                except Exception as e:
                    raise LLMError(f'Failed to create Azure OpenAI client for deployment {deployment_name}', 
                                   provider='azure', model=deployment_name, original_error=e)
        return clients

    @property
    def provider_name(self) -> str:
        return 'azure'
