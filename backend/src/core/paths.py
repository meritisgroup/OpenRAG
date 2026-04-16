import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STORAGE_DIR = os.path.join(BASE_DIR, 'storage')

CONFIG_PATH = os.path.join(DATA_DIR, 'base_config_server.json')
PROVIDERS_PATH = os.path.join(DATA_DIR, 'providers_infos.json')
MODELS_PATH = os.path.join(DATA_DIR, 'models_infos.json')
ALL_RAGS_PATH = os.path.join(DATA_DIR, 'all_rags.json')
DATABASES_DIR = os.path.join(DATA_DIR, 'databases')
QUERIES_DIR = os.path.join(DATA_DIR, 'queries')
DOCUMENTS_DIR = os.path.join(DATA_DIR, 'documents')
REPORT_PATH = os.path.join(DATA_DIR, 'report')
CUSTOM_RAGS_DIR = os.path.join(DATA_DIR, 'custom_rags')
MERGE_RAGS_DIR = os.path.join(DATA_DIR, 'merge')
