"""
Mapping des modèles nécessaires par type de RAG
Définit quels modèles sont requis et optionnels pour chaque type de RAG
"""

RAG_MODEL_REQUIREMENTS = {
    'naive': {
        'required': ['model', 'embedding_model'],
        'optional': []
    },
    'naive_chatbot': {
        'required': ['model'],
        'optional': []
    },
    'advanced_rag': {
        'required': ['model', 'embedding_model'],
        'optional': ['reranker_model']
    },
    'agentic': {
        'required': ['model', 'embedding_model'],
        'optional': ['reranker_model']
    },
    'agentic_router': {
        'required': ['model', 'embedding_model'],
        'optional': ['reranker_model']
    },
    'reranker_rag': {
        'required': ['model', 'embedding_model', 'reranker_model'],
        'optional': []
    },
    'query_reformulation': {
        'required': ['model', 'embedding_model'],
        'optional': []
    },
    'semantic_chunking': {
        'required': ['model', 'embedding_model'],
        'optional': []
    },
    'graph': {
        'required': ['model', 'embedding_model'],
        'optional': []
    },
    'corrective_rag': {
        'required': ['model', 'embedding_model'],
        'optional': []
    },
    'contextual_retrieval': {
        'required': ['model', 'embedding_model'],
        'optional': []
    },
    'query_based': {
        'required': ['model', 'embedding_model'],
        'optional': []
    },
    'merger': {
        'required': ['model', 'embedding_model'],
        'optional': ['reranker_model']
    }
}


def get_required_models_for_rag(rag_name: str, config: dict) -> dict:
    """
    Retourne la liste des modèles requis pour un type de RAG
    
    Args:
        rag_name: Nom du type de RAG
        config: Configuration serveur pour détecter les modèles optionnels utilisés
    
    Returns:
        Dict avec 'required' et 'optional' (listes de clés de modèles)
    """
    import json
    import os
    
    # Gestion des custom RAGs
    custom_rag_path = f'data/custom_rags/{rag_name}.json'
    if os.path.exists(custom_rag_path):
        with open(custom_rag_path, 'r') as f:
            custom_config = json.load(f)
        base_rag = custom_config.get('base', 'naive')
        return get_required_models_for_rag(base_rag, config)
    
    # Gestion des merge RAGs
    merge_rag_path = f'data/merge/{rag_name}.json'
    if os.path.exists(merge_rag_path):
        return {
            'required': ['model', 'embedding_model'],
            'optional': ['reranker_model']
        }
    
    # RAG standard
    requirements = RAG_MODEL_REQUIREMENTS.get(rag_name, {
        'required': ['model', 'embedding_model'],
        'optional': []
    })
    
    # Ajouter les modèles optionnels qui sont configurés
    optional_models = []
    for opt_model in requirements['optional']:
        if config.get(opt_model):
            optional_models.append(opt_model)
    
    return {
        'required': requirements['required'],
        'optional': optional_models
    }