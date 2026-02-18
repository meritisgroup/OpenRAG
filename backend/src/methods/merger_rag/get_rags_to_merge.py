def get_rag_to_merge(rag_name, config_server, models_infos, databases_name=[]):
    from factory_RagAgent import get_rag_agent, get_custom_rag_agent
    if 'base' in config_server.keys():
        agent = get_rag_agent(base_rag_name=config_server['base'], config_server=config_server, models_infos=models_infos, databases_name=databases_name)
    else:
        agent = get_rag_agent(rag_name=rag_name, config_server=config_server, models_infos=models_infos, databases_name=databases_name)
    return agent
