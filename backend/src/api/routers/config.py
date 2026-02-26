import json
import os
from fastapi import APIRouter, HTTPException

from api.schemas.config import (
    ConfigResponse, ConfigUpdateRequest, LocalParamsRequest, SystemInfo, 
    ProviderInfo, ModelInfo, ChangeConfigServerRequest, ModelsUpdateRequest, 
    ProvidersUpdateRequest
)
from factory import RAGFactory
from factory_RagAgent import change_local_parameters, put_default_local_parameters, change_config_server

router = APIRouter()

CONFIG_PATH = 'data/base_config_server.json'
PROVIDERS_PATH = 'data/providers_infos.json'
MODELS_PATH = 'data/models_infos.json'


def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def _save_json(path: str, data: dict) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


@router.get("", response_model=ConfigResponse)
def get_config():
    config = _load_json(CONFIG_PATH)
    local_params = config.get('local_params', {
        'forced_system_prompt': False,
        'generation_system_prompt_name': 'default'
    })
    return ConfigResponse(config=config, local_params=local_params)


@router.put("")
def update_config(request: ConfigUpdateRequest):
    _save_json(CONFIG_PATH, request.config)
    return {"status": "updated"}


@router.put("/local-params")
def update_local_params(request: LocalParamsRequest):
    config = _load_json(CONFIG_PATH)
    config['local_params'] = {
        'forced_system_prompt': request.forced_system_prompt,
        'generation_system_prompt_name': request.generation_system_prompt_name
    }
    _save_json(CONFIG_PATH, config)
    change_local_parameters(config['local_params'])
    return {"status": "updated"}


@router.post("/local-params/reset")
def reset_local_params():
    put_default_local_parameters()
    return {"status": "reset"}


@router.get("/system", response_model=SystemInfo)
def get_system_info():
    providers_data = _load_json(PROVIDERS_PATH)
    providers = [
        ProviderInfo(name=name, api_key=data.get('api_key'), url=data.get('url'), type=data.get('type'))
        for name, data in providers_data.items()
    ]
    
    models_data = _load_json(MODELS_PATH)
    models = [
        ModelInfo(
            name=name,
            provider=data.get('provider', ''),
            type=data.get('type', 'llm'),
            url=data.get('url'),
            api_key=data.get('api_key')
        )
        for name, data in models_data.items()
    ]
    
    databases_path = 'data/databases'
    databases = []
    if os.path.exists(databases_path):
        databases = [d for d in os.listdir(databases_path) if d != '.gitkeep']
    
    return SystemInfo(
        providers=providers,
        models=models,
        databases=databases,
        rag_methods=RAGFactory.list_available_rags()
    )


@router.put("/change-server")
def change_server_config(request: ChangeConfigServerRequest):
    config = _load_json(CONFIG_PATH)
    updated_config = change_config_server(
        rag_name=request.rag_name,
        config_server=config
    )
    _save_json(CONFIG_PATH, updated_config)
    return {"status": "updated", "config": updated_config}


@router.get("/models")
def get_models():
    return _load_json(MODELS_PATH)


@router.put("/models")
def update_models(request: ModelsUpdateRequest):
    _save_json(MODELS_PATH, request.models)
    return {"status": "updated"}


@router.get("/providers")
def get_providers():
    return _load_json(PROVIDERS_PATH)


@router.put("/providers")
def update_providers(request: ProvidersUpdateRequest):
    _save_json(PROVIDERS_PATH, request.providers)
    return {"status": "updated"}


@router.get("/all-rags")
def get_all_rags():
    path = 'data/all_rags.json'
    return _load_json(path)


@router.put("/all-rags")
def update_all_rags(request: ConfigUpdateRequest):
    path = 'data/all_rags.json'
    _save_json(path, request.config)
    return {"status": "updated"}
