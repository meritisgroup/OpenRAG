import streamlit as st
from typing import List, Dict, Any
from streamlit_.api_client import APIClient
from streamlit_.api_client.exceptions import APIError
from streamlit_.core.config import API_BASE_URL

_client = APIClient(API_BASE_URL)


def _get_metadatas(database_name: str) -> Dict[str, Any]:
    try:
        return _client.get_database_metadatas(database_name)
    except APIError as e:
        st.error(f"Error loading metadatas: {e}")
        return {'keys': [], 'documents': {}}


def _save_metadatas(database_name: str, metadatas: Dict[str, Any]) -> bool:
    try:
        _client.update_database_metadatas(database_name, metadatas)
        return True
    except APIError as e:
        st.error(f"Error saving metadatas: {e}")
        return False


def get_documents(metadatas: Dict[str, Any]) -> List[str]:
    return list(metadatas.get('documents', {}).keys())


def get_documents_of_database(database_name: str) -> List[str]:
    metadatas = _get_metadatas(database_name)
    return list(metadatas.get('documents', {}).keys())


def save_new_metadata_key(database_name: str, new_key: str) -> bool:
    metadatas = _get_metadatas(database_name)
    if 'keys' not in metadatas:
        metadatas['keys'] = []
    if new_key not in metadatas['keys']:
        metadatas['keys'].append(new_key)
    for doc in metadatas.get('documents', {}):
        metadatas['documents'][doc][new_key] = None
    return _save_metadatas(database_name, metadatas)


def get_metadatas(database_name: str) -> Dict[str, Any]:
    return _get_metadatas(database_name)


def load_metadatas_keys(database_name: str) -> List[str]:
    metadatas = _get_metadatas(database_name)
    return metadatas.get('keys', [])


def save_individual_modification(database_name: str, key: str, metadatas_inputs: Dict[str, str]) -> bool:
    metadatas = _get_metadatas(database_name)
    documents = get_documents(metadatas)
    for doc in documents:
        if metadatas_inputs.get(doc) and metadatas_inputs[doc] != '':
            metadatas['documents'][doc][key] = metadatas_inputs[doc]
    return _save_metadatas(database_name, metadatas)


def save_global_modification(database_name: str, key: str, new_input: str, metadata_input: Dict[str, bool]) -> bool:
    metadatas = _get_metadatas(database_name)
    documents = get_documents(metadatas)
    for doc in documents:
        if metadata_input.get(doc, False):
            metadatas['documents'][doc][key] = new_input
    return _save_metadatas(database_name, metadatas)


def delete_metadata(database_name: str, key: str) -> bool:
    metadatas = _get_metadatas(database_name)
    if key in metadatas.get('keys', []):
        metadatas['keys'].remove(key)
    for doc, fields in metadatas.get('documents', {}).items():
        if key in fields:
            fields.pop(key)
    return _save_metadatas(database_name, metadatas)


def add_documents_metadata(database_name: str) -> bool:
    try:
        document_list = _client.get_database_documents(database_name)
        metadatas = _get_metadatas(database_name)
        keys = metadatas.get('keys', [])
        saved_docs = list(metadatas.get('documents', {}).keys())
        for saved_doc in saved_docs:
            if saved_doc not in document_list:
                metadatas['documents'].pop(saved_doc, None)
        for doc in document_list:
            if doc not in metadatas.get('documents', {}):
                metadatas['documents'][doc] = {}
            if keys:
                for key in keys:
                    if key not in metadatas['documents'][doc]:
                        metadatas['documents'][doc][key] = None
        return _save_metadatas(database_name, metadatas)
    except APIError as e:
        st.error(f"Error updating documents metadata: {e}")
        return False


def init_metadata(database_name: str) -> bool:
    metadatas = {'keys': [], 'documents': {}}
    return _save_metadatas(database_name, metadatas)


def delete_database_metadata(database_name: str):
    pass


def delete_button():
    st.session_state.button = not st.session_state.button
