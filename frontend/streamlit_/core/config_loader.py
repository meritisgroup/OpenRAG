import json
from streamlit_.api_client import APIClient
from streamlit_.core.config import API_BASE_URL


def load_app_config(config_path: str = 'data/base_config_server.json'):
    try:
        client = APIClient(API_BASE_URL)
        response = client.get_config()
        return response.get('config', {})
    except Exception as e:
        raise RuntimeError(f'Failed to load config from API: {e}')


def load_json_config(config_path: str):
    with open(config_path, 'r') as file:
        return json.load(file)


def get_config_with_fallback(config_path: str = 'data/base_config_server.json'):
    try:
        config = load_app_config(config_path)
        return config, None
    except Exception:
        return load_json_config(config_path), None
