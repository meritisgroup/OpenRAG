import json
import os
import logging
import time
from fastapi import APIRouter, HTTPException

from api.schemas.config import (
    ConfigResponse, ConfigUpdateRequest, LocalParamsRequest, SystemInfo, 
    ProviderInfo, ModelInfo, ChangeConfigServerRequest, ModelsUpdateRequest, 
    ProvidersUpdateRequest
)
from factory import RAGFactory
from factory_RagAgent import change_local_parameters, put_default_local_parameters, change_config_server

logger = logging.getLogger(__name__)

router = APIRouter()

from core.paths import CONFIG_PATH, PROVIDERS_PATH, MODELS_PATH, ALL_RAGS_PATH, DATABASES_DIR


def _mask_secret(value: str) -> str:
    if not value or len(value) < 8:
        return '***' if value else ''
    return value[:4] + '***' + value[-4:]


def _mask_dict_secrets(data: dict, key_names=('api_key',)) -> dict:
    masked = {}
    for k, v in data.items():
        if isinstance(v, dict):
            masked[k] = _mask_dict_secrets(v, key_names)
        elif k in key_names and isinstance(v, str):
            masked[k] = _mask_secret(v)
        else:
            masked[k] = v
    return masked


def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def _save_json(path: str, data: dict) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def _test_model_availability(model_name: str, model_info: dict, timeout: int = 20) -> dict:
    """
    Teste si un modèle est disponible en faisant une requête réelle

    Args:
        model_name: Nom du modèle à tester
        model_info: Informations du modèle (url, api_key, type)
        timeout: Timeout en secondes

    Returns:
        dict: {'available': bool, 'error': Optional[str]}
    """
    max_retries = 2
    last_result = None

    for attempt in range(max_retries):
        result = _test_model_availability_once(model_name, model_info, timeout)
        if result['available']:
            return result
        last_result = result
        error_msg = (last_result.get('error') or '').lower()
        is_transient = any(kw in error_msg for kw in [
            'connection', 'timeout', 'connexion', 'timed out',
            'connectionrefused', 'connectionerror', 'network',
        ])
        if not is_transient or attempt == max_retries - 1:
            break
        wait = 2 ** attempt
        logger.info("Retry %d/%d for model '%s' after transient error: %s (waiting %ds)",
                     attempt + 1, max_retries, model_name, last_result['error'], wait)
        time.sleep(wait)

    return last_result


def _test_model_availability_once(model_name: str, model_info: dict, timeout: int = 20) -> dict:
    import requests
    from openai import OpenAI, AzureOpenAI, APIError, APIConnectionError

    try:
        url = model_info.get('url')
        api_key = model_info.get('api_key', '')
        model_type = model_info.get('type', 'llm')
        provider = model_info.get('provider', 'openai').lower()

        if provider == 'azure':
            api_version = model_info.get('api_version', '2024-02-01')
            client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=url, timeout=timeout)
        else:
            if url:
                url = url.rstrip('/')
                if not any(url.endswith(s) for s in ('/v1', '/v2', '/v3', '/v4', '/v1beta/openai', '/openai/v1', '/inference/v1', '/compatible-mode/v1', '/studio/v1', '/v3/openai', '/api/v1', '/api/v3')):
                    url = url + '/v1'
                base_url = url
            else:
                base_url = None
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        # Vérifier d'abord si le modèle existe dans la liste des modèles disponibles
        # SAUF pour les rerankers (ils n'ont généralement pas l'endpoint /v1/models)
        if model_type != 'reranker':
            try:
                available_models = client.models.list()
                available_model_names = [m.id for m in available_models.data]

                # Vérifier si le modèle demandé est dans la liste (recherche exacte ou partielle)
                model_exists = (
                    model_name in available_model_names or
                    any(model_name.lower() in m.lower() or m.lower() in model_name.lower() for m in available_model_names)
                )

                if not model_exists:
                    models_list_str = ', '.join(available_model_names[:10])
                    if len(available_model_names) > 10:
                        models_list_str += '...'
                    return {'available': False, 'error': f'Modèle "{model_name}" non trouvé sur le serveur. Modèles disponibles: {models_list_str}'}
            except Exception as e:
                logger.debug(f"Model list retrieval failed for {model_name}: {e}")
                pass

        if model_type == 'embedding':
            response = client.embeddings.create(input="test", model=model_name)
            return {'available': True, 'error': None}
        elif model_type == 'reranker':
            _rerank_suffixes = ('/v1', '/v2', '/v3', '/v4', '/v1beta/openai', '/openai/v1', '/inference/v1', '/compatible-mode/v1', '/studio/v1', '/v3/openai', '/api/v1', '/api/v3')
            rerank_base = url + '/v1' if url and not any(url.endswith(s) for s in _rerank_suffixes) else (url or '')
            rerank_url = rerank_base + '/rerank'
            payload = {'model': model_name, 'query': 'test', 'documents': ['test']}
            response = requests.post(rerank_url, json=payload, timeout=timeout)

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    if 'results' in response_data or ('data' in response_data and isinstance(response_data['data'], list)):
                        return {'available': True, 'error': None}
                    else:
                        return {'available': False, 'error': 'Réponse invalide du serveur rerank (format incorrect)'}
                except Exception as e:
                    logger.debug(f"Failed to parse rerank response JSON: {e}")
                    return {'available': False, 'error': 'Réponse invalide du serveur rerank'}
            else:
                error_detail = f'Code HTTP {response.status_code}'
                try:
                    error_json = response.json()
                    if 'error' in error_json:
                        error_detail = error_json['error']
                    elif 'message' in error_json:
                        error_detail = error_json['message']
                except Exception as e:
                    logger.debug(f"Failed to parse error response JSON: {e}")
                    pass

                if response.status_code == 404:
                    return {'available': False, 'error': 'Endpoint /v1/rerank non trouvé (pas un serveur de reranking)'}
                elif 'model' in str(error_detail).lower() and ('not found' in str(error_detail).lower() or 'not support' in str(error_detail).lower()):
                    return {'available': False, 'error': f'Modèle "{model_name}" non disponible pour le reranking'}
                else:
                    return {'available': False, 'error': f'Erreur du serveur rerank: {error_detail}'}
        else:
            params = {
                'model': model_name,
                'messages': [{"role": "user", "content": "test"}]
            }
            response = client.beta.chat.completions.parse(**params)
            return {'available': True, 'error': None}
    except APIConnectionError as e:
        return {'available': False, 'error': f'Erreur de connexion: {str(e)}'}
    except APIError as e:
        error_msg = str(e).lower()
        if 'model' in error_msg and ('not found' in error_msg or 'does not exist' in error_msg or 'invalid' in error_msg):
            return {'available': False, 'error': f'Modèle "{model_name}" non disponible sur ce serveur'}
        return {'available': False, 'error': f'Erreur API: {str(e)}'}
    except requests.exceptions.RequestException as e:
        error_msg = str(e).lower()
        if 'model' in error_msg and ('not found' in error_msg or 'does not exist' in error_msg):
            return {'available': False, 'error': f'Modèle "{model_name}" non disponible sur ce serveur'}
        return {'available': False, 'error': f'Erreur HTTP: {str(e)}'}
    except Exception as e:
        return {'available': False, 'error': f'Erreur inattendue: {str(e)}'}


def _validate_rag_models(rag_name: str, config: dict, models_infos: dict, timeout: int = 20) -> dict:
    """
    Valide les modèles nécessaires pour un type de RAG
    
    Args:
        rag_name: Nom du type de RAG
        config: Configuration serveur
        models_infos: Informations sur les modèles disponibles
        timeout: Timeout en secondes pour tester chaque modèle
    
    Returns:
        dict: {
            'all_available': bool,
            'models': dict,  # Résultats par clé de modèle
            'errors': list[str]
        }
    """
    from utils.rag_model_requirements import get_required_models_for_rag
    
    requirements = get_required_models_for_rag(rag_name, config)
    
    # Combiner requis et optionnels configurés
    models_to_check = requirements['required'] + requirements['optional']
    
    results = {
        'all_available': True,
        'models': {},
        'errors': []
    }
    
    for model_key in models_to_check:
        model_name = config.get(model_key)
        
        if not model_name:
            results['models'][model_key] = {
                'name': None,
                'available': False,
                'error': f"Modèle '{model_key}' non configuré"
            }
            results['all_available'] = False
            results['errors'].append(f"Modèle '{model_key}' non configuré")
            continue
        
        if model_name not in models_infos:
            results['models'][model_key] = {
                'name': model_name,
                'available': False,
                'error': f"Modèle '{model_name}' non trouvé dans models_infos.json"
            }
            results['all_available'] = False
            results['errors'].append(f"Modèle '{model_name}' non trouvé dans models_infos.json")
            continue
        
        model_info = models_infos[model_name]
        
        try:
            test_result = _test_model_availability(model_name, model_info, timeout)
        except Exception as e:
            test_result = {'available': False, 'error': f'Erreur lors du test de connexion: {str(e)}'}
        
        results['models'][model_key] = {
            'name': model_name,
            'available': test_result['available'],
            'error': test_result['error']
        }
        
        if not test_result['available']:
            results['all_available'] = False
            results['errors'].append(f"Modèle '{model_name}' ({model_key}): {test_result['error']}")
    
    return results


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
    config = request.config
    
    local_params = config.get('local_params', {})
    prompt_name = local_params.get('generation_system_prompt_name', 'default')
    all_prompts = config.get('all_system_prompt', {})
    
    if prompt_name not in all_prompts and prompt_name != 'default':
        config['local_params']['generation_system_prompt_name'] = 'default'
    
    _save_json(CONFIG_PATH, config)
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
        ProviderInfo(name=name, api_key=_mask_secret(data.get('api_key')), url=data.get('url'), type=data.get('type'))
        for name, data in providers_data.items()
    ]
    
    models_data = _load_json(MODELS_PATH)
    models = [
        ModelInfo(
            name=name,
            provider=data.get('provider', ''),
            type=data.get('type', 'llm'),
            url=data.get('url'),
            api_key=_mask_secret(data.get('api_key'))
        )
        for name, data in models_data.items()
    ]
    
    databases_path = DATABASES_DIR
    databases = []
    if os.path.exists(databases_path):
        databases = [d for d in os.listdir(databases_path) if d != '.gitkeep']
    
    return SystemInfo(
        providers=providers,
        models=models,
        databases=databases,
        rag_methods=RAGFactory.list_available_rags()
    )


@router.post("/models/test")
def test_configured_models():
    """
    Fait une requête bidon à chaque modèle configuré pour vérifier s'il est disponible
    """
    config = _load_json(CONFIG_PATH)
    models_infos = _load_json(MODELS_PATH)

    results = {
        'model': None,
        'embedding_model': None,
        'reranker_model': None,
        'model_for_image': None
    }

    model_keys = ['model', 'embedding_model', 'reranker_model', 'model_for_image']

    for key in model_keys:
        model_name = config.get(key)
        if not model_name or model_name not in models_infos:
            results[key] = {
                'name': model_name,
                'available': False,
                'error': 'Non configuré ou non trouvé dans models_infos.json'
            }
            continue

        model_info = models_infos[model_name]
        model_type = model_info.get('type', 'llm')

        test_result = _test_model_availability(model_name, model_info, timeout=10)

        result = {
            'name': model_name,
            'available': test_result['available'],
        }
        if test_result['available']:
            result['type'] = model_type
        else:
            result['error'] = test_result['error']
        results[key] = result

    return results


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
    return _mask_dict_secrets(_load_json(MODELS_PATH))


@router.put("/models")
def update_models(request: ModelsUpdateRequest):
    _save_json(MODELS_PATH, request.models)
    return {"status": "updated"}


@router.get("/providers")
def get_providers():
    return _mask_dict_secrets(_load_json(PROVIDERS_PATH))


@router.put("/providers")
def update_providers(request: ProvidersUpdateRequest):
    _save_json(PROVIDERS_PATH, request.providers)
    return {"status": "updated"}


@router.get("/all-rags")
def get_all_rags():
    return _load_json(ALL_RAGS_PATH)


@router.put("/all-rags")
def update_all_rags(request: ConfigUpdateRequest):
    _save_json(ALL_RAGS_PATH, request.config)
    return {"status": "updated"}
