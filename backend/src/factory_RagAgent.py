import json
from factory import RAGFactory
from factory.rag_registry import RAG_REGISTRY

_factory = RAGFactory()


def change_config_server(rag_name, config_server):
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
