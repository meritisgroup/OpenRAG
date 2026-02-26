import re

def get_name(rag_name, config_server, additionnal_name=''):
    if type(config_server['embedding_model']) == str:
        embedding_models = [config_server['embedding_model']]
    else:
        embedding_models = config_server['embedding_model']
    names = []
    for embedding_model in embedding_models:
        name = '{}_'.format(re.sub('[\\\\/:*?"<>|]', '_', embedding_model))
        if rag_name == 'naive' or rag_name == 'crag' or rag_name == 'reranker_rag' or (rag_name == 'query_reformulation_rag') or (rag_name == 'self') or (rag_name == 'main') or (rag_name == 'naive_chatbot') or (rag_name == 'agentic') or (rag_name == 'agentic_router'):
            name = name + '{}_{}'.format(config_server['type_retrieval'], config_server['TextSplitter'])
        elif rag_name == 'graph':
            name = 'graph_rag_{}_{}_{}'.format(name, config_server['type_retrieval'], config_server['TextSplitter'])
        elif rag_name == 'query_based':
            name = 'query_rag_{}_{}_{}'.format(name, config_server['type_retrieval'], config_server['TextSplitter'])
        elif rag_name == 'advanced_rag':
            name = name + '{}_{}'.format(config_server['type_retrieval'], config_server['TextSplitter'])
            if len(config_server['ProcessorChunks']) > 0:
                for i in range(len(config_server['ProcessorChunks'])):
                    name += '_{}'.format(config_server['ProcessorChunks'][i])
        elif rag_name == 'semantic_chunking':
            name = '{}_{}_{}'.format(name, config_server['type_retrieval'], 'Semantic_TextSplitter')
        elif rag_name == 'contextual_retrieval':
            name = '{}_{}_{}_{}'.format(name, config_server['type_retrieval'], config_server['TextSplitter'], 'contextual')
        name += '_{}'.format(config_server['data_preprocessing'])
        name += '_{}'.format(config_server['chunk_length'])
        if additionnal_name != '':
            name += '_' + additionnal_name
        name = name.lower()
        names.append(name)
    return names