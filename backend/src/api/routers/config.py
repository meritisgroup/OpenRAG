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


def _test_model_availability(model_name: str, model_info: dict, timeout: int = 10) -> dict:
    """
    Teste si un modèle est disponible en faisant une requête réelle

    Args:
        model_name: Nom du modèle à tester
        model_info: Informations du modèle (url, api_key, type)
        timeout: Timeout en secondes

    Returns:
        dict: {'available': bool, 'error': Optional[str]}
    """
    import requests
    from openai import OpenAI, APIError, APIConnectionError

    try:
        url = model_info.get('url')
        api_key = model_info.get('api_key', '')
        model_type = model_info.get('type', 'llm')

        if url:
            base_url = url + '/v1' if not url.endswith('/v1') else url
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
            except Exception:
                # Si la liste des modèles échoue, on continue avec le test normal
                pass

        if model_type == 'embedding':
            response = client.embeddings.create(input="test", model=model_name)
            return {'available': True, 'error': None}
        elif model_type == 'reranker':
            rerank_url = url + '/v1/rerank'
            payload = {'model': model_name, 'query': 'test', 'documents': ['test']}
            response = requests.post(rerank_url, json=payload, timeout=timeout)

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    if 'results' in response_data or ('data' in response_data and isinstance(response_data['data'], list)):
                        return {'available': True, 'error': None}
                    else:
                        return {'available': False, 'error': 'Réponse invalide du serveur rerank (format incorrect)'}
                except Exception:
                    return {'available': False, 'error': 'Réponse invalide du serveur rerank'}
            else:
                error_detail = f'Code HTTP {response.status_code}'
                try:
                    error_json = response.json()
                    if 'error' in error_json:
                        error_detail = error_json['error']
                    elif 'message' in error_json:
                        error_detail = error_json['message']
                except:
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
                'messages': [{"role": "user", "content": "test"}],
                'max_tokens': 5
            }
            params.pop('impacts', None)
            response = client.chat.completions.create(**params)
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


def _validate_rag_models(rag_name: str, config: dict, models_infos: dict, timeout: int = 10) -> dict:
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


@router.post("/models/test")
def test_configured_models():
    """
    Fait une requête bidon à chaque modèle configuré pour vérifier s'il est disponible
    """
    import requests
    from openai import OpenAI, APIError, APIConnectionError

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
        url = model_info.get('url')
        api_key = model_info.get('api_key', '')

        try:
            if url:
                base_url = url + '/v1' if not url.endswith('/v1') else url
            else:
                base_url = None
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=10)
        except Exception as e:
            results[key] = {
                'name': model_name,
                'available': False,
                'error': f'Erreur création client: {str(e)}'
            }
            continue

        try:
            # Vérifier d'abord si le modèle existe dans la liste des modèles disponibles
            model_exists = False
            available_model_names = []
            try:
                available_models = client.models.list()
                available_model_names = [m.id for m in available_models.data]

                # Vérifier si le modèle demandé est dans la liste (recherche exacte ou partielle)
                model_exists = (
                    model_name in available_model_names or
                    any(model_name.lower() in m.lower() or m.lower() in model_name.lower() for m in available_model_names)
                )

                if not model_exists:
                    results[key] = {
                        'name': model_name,
                        'available': False,
                        'error': f'Modèle "{model_name}" non trouvé sur le serveur. Modèles disponibles: {available_model_names[:10]}...' if len(available_model_names) > 10 else f'Modèle "{model_name}" non trouvé sur le serveur. Modèles disponibles: {available_model_names}'
                    }
                    continue
            except Exception as e:
                # Si la liste des modèles échoue, on continue avec le test normal
                pass

            if model_type == 'embedding':
                response = client.embeddings.create(
                    input="test",
                    model=model_name
                )
                results[key] = {
                    'name': model_name,
                    'available': True,
                    'type': model_type
                }
            elif model_type == 'reranker':
                rerank_url = url + '/v1/rerank'
                payload = {
                    'model': model_name,
                    'query': 'test query',
                    'documents': ['test document']
                }
                response = requests.post(rerank_url, json=payload, timeout=10)

                # Vérifier que la réponse est valide et contient des résultats de reranking
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        # Vérifier que la réponse contient les champs attendus pour un reranker
                        if 'results' in response_data or ('data' in response_data and isinstance(response_data['data'], list)):
                            results[key] = {
                                'name': model_name,
                                'available': True,
                                'type': model_type
                            }
                        else:
                            # Réponse reçue mais pas le format attendu pour un reranker
                            results[key] = {
                                'name': model_name,
                                'available': False,
                                'error': f'Réponse invalide du serveur rerank (format de réponse incorrect). Vérifiez que l\'URL pointe vers un serveur de reranking'
                            }
                    except Exception:
                        # Réponse reçue mais pas du JSON valide
                        results[key] = {
                            'name': model_name,
                            'available': False,
                            'error': f'Réponse invalide du serveur rerank. Vérifiez que l\'URL pointe vers un serveur de reranking'
                        }
                else:
                    # Le serveur a répondu avec un code d'erreur
                    error_detail = f'Code HTTP {response.status_code}'
                    try:
                        error_json = response.json()
                        if 'error' in error_json:
                            error_detail = error_json['error']
                        elif 'message' in error_json:
                            error_detail = error_json['message']
                    except:
                        pass

                    # Vérifier si l'erreur indique que le endpoint rerank n'existe pas
                    if response.status_code == 404:
                        results[key] = {
                            'name': model_name,
                            'available': False,
                            'error': f'Endpoint /v1/rerank non trouvé. Cette URL ne semble pas être un serveur de reranking'
                        }
                    elif 'model' in str(error_detail).lower() and ('not found' in str(error_detail).lower() or 'not support' in str(error_detail).lower()):
                        results[key] = {
                            'name': model_name,
                            'available': False,
                            'error': f'Modèle "{model_name}" non disponible pour le reranking sur ce serveur'
                        }
                    else:
                        results[key] = {
                            'name': model_name,
                            'available': False,
                            'error': f'Erreur du serveur rerank: {error_detail}'
                        }
            else:
                params = {
                    'model': model_name,
                    'messages': [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "test"}
                    ],
                    'max_tokens': 5
                }
                params.pop('impacts', None)
                response = client.chat.completions.create(**params)
                results[key] = {
                    'name': model_name,
                    'available': True,
                    'type': model_type
                }
        except APIConnectionError as e:
            results[key] = {
                'name': model_name,
                'available': False,
                'error': f'Erreur de connexion: {str(e)}'
            }
        except APIError as e:
            error_msg = str(e).lower()
            # Vérifier si l'erreur indique que le modèle n'existe pas
            if 'model' in error_msg and ('not found' in error_msg or 'does not exist' in error_msg or 'invalid' in error_msg):
                results[key] = {
                    'name': model_name,
                    'available': False,
                    'error': f'Modèle "{model_name}" non disponible sur ce serveur'
                }
            else:
                results[key] = {
                    'name': model_name,
                    'available': False,
                    'error': f'Erreur API (code {e.status_code}): {str(e)}'
                }
        except requests.exceptions.RequestException as e:
            error_msg = str(e).lower()
            # Vérifier si l'erreur indique que le modèle n'existe pas
            if 'model' in error_msg and ('not found' in error_msg or 'does not exist' in error_msg):
                results[key] = {
                    'name': model_name,
                    'available': False,
                    'error': f'Modèle "{model_name}" non disponible sur ce serveur'
                }
            else:
                results[key] = {
                    'name': model_name,
                    'available': False,
                    'error': f'Erreur HTTP: {str(e)}'
                }
        except Exception as e:
            results[key] = {
                'name': model_name,
                'available': False,
                'error': f'Erreur inattendue: {str(e)}'
            }

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
