from pydantic import BaseModel
from typing import Dict, Any, Optional, List


class ConfigResponse(BaseModel):
    config: Dict[str, Any]
    local_params: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any]


class LocalParamsRequest(BaseModel):
    forced_system_prompt: bool = False
    generation_system_prompt_name: str = 'default'


class ProviderInfo(BaseModel):
    name: str
    api_key: Optional[str] = None
    url: Optional[str] = None
    type: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    provider: str
    type: str
    url: Optional[str] = None
    api_key: Optional[str] = None


class SystemInfo(BaseModel):
    providers: List[ProviderInfo]
    models: List[ModelInfo]
    databases: List[str]
    rag_methods: List[str]


class ChangeConfigServerRequest(BaseModel):
    rag_name: Optional[str] = None
    mode: str = 'Simple'


class ModelsUpdateRequest(BaseModel):
    models: Dict[str, Any]


class ProvidersUpdateRequest(BaseModel):
    providers: Dict[str, Any]
