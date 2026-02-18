from dotenv import load_dotenv, find_dotenv, set_key
import os

from streamlit_.api_client import APIClient
from streamlit_.api_client.exceptions import APIError
from streamlit_.core.config import API_BASE_URL

_client = APIClient(API_BASE_URL)


def modify_env(key, value):
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    set_key(dotenv_path, key, str(value))
    os.environ[key] = str(value)


def get_possible_embeddings_model(provider):
    if provider == 'ollama':
        return ['mxbai-embed-large:latest', 'bge-m3:latest', 'all-minilm:22m']
    elif provider == 'openai':
        return ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']
    elif provider == 'vllm':
        return ['BAAI/bge-m3', 'mixedbread-ai/mxbai-embed-large-v1', 'Qwen/Qwen3-Embedding-0.6B', 'sentence-transformers/all-MiniLM-L6-v2']
    return []


def get_default_embeddings_model(provider):
    models = get_possible_embeddings_model(provider=provider)
    return models[0] if models else None


def get_config_rag(rag_name):
    custom_rags_name = get_custom_rags_name()
    merge_rags_name = get_merge_rags_name()
    
    try:
        if rag_name in custom_rags_name:
            return _client.get_custom_rag(rag_name)
        elif rag_name in merge_rags_name:
            all_rags = _client.get_all_rags()
            if rag_name in all_rags:
                return _client.get_custom_rag(rag_name)
    except APIError:
        pass
    
    try:
        config_response = _client.get_config()
        return config_response.get('config', {})
    except APIError:
        return {}


def get_custom_rags_name():
    try:
        response = _client.list_custom_rags()
        return response.get('custom_rags', [])
    except APIError:
        return []


def get_merge_rags_name():
    try:
        response = _client.list_merge_rags()
        return response.get('merge_rags', [])
    except APIError:
        return []
