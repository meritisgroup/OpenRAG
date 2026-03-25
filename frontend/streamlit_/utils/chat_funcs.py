import streamlit as st
import re
from streamlit_.services import RAGService


def get_model_error_message(model_key: str, model_name: str, error_msg: str, rag_method: str):
    """
    Retourne un message d'erreur explicite selon le type de modèle.
    Returns: (error_title, error_detail)
    """
    messages = {
        'reranker_model': (
            f"Le reranker '{model_name}' n'est pas disponible",
            f"Le reranker est obligatoire pour {rag_method}. Vérifiez que votre serveur de reranking est accessible et que le modèle '{model_name}' est disponible."
        ),
        'model': (
            f"Le modèle LLM '{model_name}' n'est pas disponible",
            f"Le modèle de langage est obligatoire. Vérifiez que votre serveur LLM est accessible et que le modèle '{model_name}' est disponible."
        ),
        'embedding_model': (
            f"Le modèle d'embedding '{model_name}' n'est pas disponible",
            f"Le modèle d'embedding est obligatoire. Vérifiez que votre serveur d'embedding est accessible et que le modèle '{model_name}' est disponible."
        ),
        'model_for_image': (
            f"Le modèle de vision '{model_name}' n'est pas disponible",
            f"Le modèle pour les images est configuré mais non disponible. Vérifiez que le modèle '{model_name}' est accessible."
        )
    }
    
    return messages.get(model_key, (
        f"Le modèle '{model_name}' ({model_key}) n'est pas disponible",
        error_msg
    ))


def get_chat_agent(rag_method, databases_name, session_state=None, validate_models: bool = True, create_new_session: bool = False):
    """
    Crée un agent RAG avec validation optionnelle des modèles
    
    Args:
        rag_method: Type de RAG à créer
        databases_name: Liste des bases de données
        session_state: État de session (optionnel)
        validate_models: Si True, valide les modèles avant création (défaut: True)
        create_new_session: Si True, crée une nouvelle session (utile pour multi-RAG)
    
    Returns:
        session_id: ID de session pour l'agent créé
    
    Raises:
        APIError: Si la validation échoue ou si erreur lors de la création
    """
    from streamlit_.api_client.exceptions import APIError
    
    if session_state is None:
        session_state = st.session_state
    
    client = session_state.get('api_client')
    if client:
        RAGService.set_client(client)
    
    config = session_state['config_server'].copy()

    try:
        if 'custom_rags' in session_state.keys() and rag_method in session_state['custom_rags']:
            custom_config = client.get_custom_rag(rag_method)
            custom_config['params_host_llm'] = config['params_host_llm']
            base_rag = custom_config.get('base', rag_method)
            session_id = RAGService.get_chat_agent(
                rag_method=base_rag,
                databases_name=databases_name,
                config_server=custom_config,
                models_infos=session_state['models_infos'],
                validate_models=validate_models,
                create_new_session=create_new_session
            )
            session_state['active_rag_config'] = custom_config.copy()
        elif 'merge_rags' in session_state.keys() and rag_method in session_state['merge_rags']:
            merge_config = client.get_merge_rag(rag_method)
            merge_config['params_host_llm'] = config['params_host_llm']
            session_id = RAGService.get_chat_agent(
                rag_method='merger',
                databases_name=databases_name,
                config_server=merge_config,
                models_infos=session_state['models_infos'],
                validate_models=validate_models,
                create_new_session=create_new_session
            )
            session_state['active_rag_config'] = merge_config.copy()
        else:
            session_id = RAGService.get_chat_agent(
                rag_method=rag_method,
                databases_name=databases_name,
                config_server=config,
                models_infos=session_state['models_infos'],
                validate_models=validate_models,
                create_new_session=create_new_session
            )
            session_state['active_rag_config'] = config.copy()
        
        session_state['api_session_id'] = session_id
        return session_id
        
    except APIError as e:
        # Gérer les erreurs de validation de modèles
        error_data = e.args[0] if e.args else {}
        
        if isinstance(error_data, dict):
            # L'erreur peut être wrappée dans 'detail' (FastAPI)
            detail_data = error_data.get('detail', error_data)
            
            if isinstance(detail_data, dict) and 'validation' in detail_data:
                st.error("⚠️ Certains modèles nécessaires ne sont pas disponibles :")
                validation = detail_data['validation']
                
                for model_key, result in validation.get('models', {}).items():
                    if not result.get('available', False):
                        model_name = result.get('name', 'N/A')
                        error_msg = result.get('error', 'Erreur inconnue')
                        
                        title, detail_msg = get_model_error_message(model_key, model_name, error_msg, rag_method)
                        st.error(f"❌ {title}")
                        st.error(f"   {detail_msg}")
                
                # Afficher un résumé
                st.warning("💡 Veuillez vérifier la disponibilité des modèles dans la page de configuration")
                st.stop()
            elif isinstance(detail_data, dict) and 'error' in detail_data:
                # Autre erreur avec message
                st.error(f"❌ Erreur: {detail_data['error']}")
                st.stop()
            else:
                # Autre erreur sans message explicite
                st.error(f"❌ Erreur lors de la création du RAG: {error_data}")
                st.stop()
        else:
            # Erreur non structurée
            st.error(f"❌ Erreur lors de la création du RAG: {str(e)}")
            st.stop()


def change_default_prompt():
    try:
        default_value = st.session_state['config_server']['local_params']['generation_system_prompt_name']
        st.session_state['system_prompt_selected'] = default_value
    except Exception as e:
        st.session_state['system_prompt_selected'] = 'default'


def handle_click():
    st.session_state['button_clicked'] = True


def reset_success_button():
    st.session_state['success'] = False


def clean_markdown(text: str) -> str:
    return re.sub('\\*{1,2}', '', text)


def prepare_show_context(chunks):
    blocks = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            cleaned_context = clean_markdown(chunk.get('text', ''))
            block = ''
            if chunk.get('document'):
                block += f"source : {chunk['document']}\n\n"
            block += f'{cleaned_context}'
            if chunk.get('rerank_score') is not None:
                block += f"\n\n Rerank score : {chunk['rerank_score']}"
        else:
            cleaned_context = clean_markdown(chunk.text if hasattr(chunk, 'text') else str(chunk))
            block = ''
            if hasattr(chunk, 'document') and chunk.document is not None:
                block += f"source : {chunk.document}\n\n"
            block += f'{cleaned_context}'
            if hasattr(chunk, 'rerank_score') and chunk.rerank_score is not None:
                block += f"\n\n Rerank score : {chunk.rerank_score}"
        blocks.append(block)
    separator = '\n' + '-' * 200 + '\n'
    output = separator.join(blocks)
    return output
