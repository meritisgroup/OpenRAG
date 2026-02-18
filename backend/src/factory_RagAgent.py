import json
from factory import RAGFactory
from factory.rag_registry import RAG_REGISTRY

_factory = RAGFactory()


def change_config_server(rag_name, config_server, mode):
    if mode == 'Simple':
        if config_server['default_mode_provider'] == 'ollama':
            config_server['model'] = 'gemma3:12b'
            config_server['embedding_model'] = 'mxbai-embed-large:latest'
            config_server['reranker_model'] = 'gemma3:12b'
            config_server['model_for_image'] = 'gemma3:12b'
        elif config_server['default_mode_provider'] == 'vllm':
            config_server['model'] = 'google/gemma-3-12b-it'
            config_server['embedding_model'] = 'BAAI/bge-m3'
            config_server['reranker_model'] = 'BAAI/bge-reranker-v2-m3'
            config_server['model_for_image'] = 'google/gemma-3-12b-it'
        elif config_server['default_mode_provider'] == 'openai':
            config_server['model'] = 'gpt-4o-mini'
            config_server['embedding_model'] = 'text-embedding-3-small'
            config_server['reranker_model'] = 'gpt-4.1-nano-2025-04-14'
            config_server['model_for_image'] = 'gpt-4.1-mini'
        elif config_server['default_mode_provider'] == 'mistral':
            config_server['model'] = 'mistral-small-2503'
            config_server['embedding_model'] = 'mistral-embed'
            config_server['reranker_model'] = 'mistral-small-2503'
            config_server['model_for_image'] = 'mistral-small-2503'
        if rag_name not in ('copali', 'vlm'):
            if config_server['type_retrieval'] not in ['embeddings', 'bm25', 'hybrid']:
                config_server['type_retrieval'] = 'embeddings'
    return config_server


def change_local_parameters(custom_local_config):
    config_path = 'data/base_config_server.json'
    with open(config_path, 'r') as file:
        config = json.load(file)
        config['local_params'] = custom_local_config
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)


def put_default_local_parameters():
    default_local_config = {'forced_system_prompt': False, 'generation_system_prompt_name': 'default'}
    config_path = 'data/base_config_server.json'
    with open(config_path, 'r') as file:
        config = json.load(file)
        config['local_params'] = default_local_config
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)


def get_rag_agent(rag_name, config_server, models_infos, databases_name=['']):
    return _factory.get_agent(
        rag_name=rag_name,
        config_server=config_server,
        models_infos=models_infos,
        databases_name=databases_name,
        custom=False
    )


def get_custom_rag_agent(base_rag_name, config_server, models_infos, databases_name=['']):
    return _factory.get_agent(
        rag_name=base_rag_name,
        config_server=config_server,
        models_infos=models_infos,
        databases_name=databases_name,
        custom=True
    )


def list_available_rags():
    return RAGFactory.list_available_rags()
