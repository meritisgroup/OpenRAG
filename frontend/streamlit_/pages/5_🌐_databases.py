import streamlit as st
import pandas as pd
import re
import zipfile
import io
from streamlit_.api_client import APIClient
from streamlit_.api_client.exceptions import APIError
from streamlit_.core.config import API_BASE_URL

_client = APIClient(API_BASE_URL)

if 'databases' not in st.session_state:
    st.session_state['databases'] = {}
if 'new_doc' not in st.session_state:
    st.session_state['new_doc'] = False
if 'processing_warnings' not in st.session_state:
    st.session_state['processing_warnings'] = []
if 'document_deleted' not in st.session_state:
    st.session_state['document_deleted'] = False

st.markdown('# Database Manager')
st.markdown('## Database creation')
left, right = st.columns([6, 1], vertical_alignment='bottom')
st.session_state['databases']['new_name'] = left.text_input(label='**Database name**', placeholder='Database name: alphanumeric characters, underscores and hyphens only', label_visibility='collapsed')
if right.button(label='Create DataBase', type='primary', use_container_width=True):
    if st.session_state['databases']['new_name'] in st.session_state.get('all_databases', []):
        st.error(f"Database named {st.session_state['databases']['new_name']} already exists", icon='🚨')
    elif not bool(re.fullmatch('^[a-z0-9_-]+$', st.session_state['databases']['new_name'])):
        st.error('Invalid RAG name, only alphanumeric characters, underscores, lowercase letters and hyphens are allowed', icon='🚨')
    else:
        try:
            _client.create_database(st.session_state['databases']['new_name'])
            _client.update_database_metadatas(st.session_state['databases']['new_name'], {'documents': []})
            st.session_state['all_databases'].append(st.session_state['databases']['new_name'])
            st.rerun()
        except APIError as e:
            st.error(f"Error creating database: {e}")

st.markdown('## Database visualization')
left, right = st.columns([6, 1], vertical_alignment='bottom')
all_databases = st.session_state.get('all_databases', [])
database_name = left.selectbox(label='Choose database', options=all_databases, label_visibility='collapsed') if all_databases else None
if right.button(label='Delete DataBase', type='primary', use_container_width=True) and database_name:
    db_name_to_delete = database_name
    try:
        _client.delete_database(db_name_to_delete)
        _client.delete_storage_by_prefix(db_name_to_delete)
        st.session_state['all_databases'].remove(db_name_to_delete)
        if st.session_state.get('chat_database_name') == db_name_to_delete:
            st.session_state['chat_database_name'] = None
        benchmark_db = st.session_state.get('benchmark_database', [])
        if db_name_to_delete in benchmark_db:
            benchmark_db.remove(db_name_to_delete)
            st.session_state['benchmark_database'] = benchmark_db
        database_name = None
        st.rerun()
    except APIError as e:
        st.error(f"Error deleting database: {e}")

document_list = []
if database_name is not None:
    try:
        document_list = _client.get_database_documents(database_name)
        if document_list:
            display_db = pd.DataFrame(data={'Doc Name': document_list})
            st.write(display_db)
        else:
            st.info('No documents in this database.')
    except APIError as e:
        st.error(f"Error loading documents: {e}")

if st.session_state['processing_warnings']:
    for msg in st.session_state['processing_warnings']:
        st.warning(msg)
    if st.button('Dismiss all warnings'):
        st.session_state['processing_warnings'] = []
        st.rerun()

if document_list and database_name:
    file_to_delete = st.selectbox('Select a document to delete:', document_list)
    if st.button('Delete selected document'):
        try:
            _client.delete_database_document(database_name, file_to_delete)
            st.success(f'Deleted {file_to_delete}')
            st.session_state['document_deleted'] = True
            st.rerun()
        except APIError as e:
            st.error(f"Error deleting document: {e}")
else:
    st.info('No documents to display or delete.')

if st.session_state['document_deleted']:
    st.success('A document has been deleted. Don\'t forget to reset the indexing!')
    st.session_state['document_deleted'] = False

SUPPORTED = ['pdf', 'docx', 'xlsx', 'txt', 'pptx', 'zip']


def new_doc():
    st.session_state['new_doc'] = True


def process_file_via_api(file_name: str, file_data: bytes, db_name: str):
    try:
        result = _client.process_document(file_name, file_data)
        if not result.get('is_valid', False):
            return False, f'Doc {file_name} was not added to database because it is empty.  \n  Please note that images are not read, even if they are in a supported filetype'
        
        _client.upload_database_document(db_name, file_name, file_data)
        return True, None
    except APIError as e:
        return False, f'Error processing {file_name}: {str(e)}'


if database_name:
    uploaded_files = st.file_uploader('**Add document to DataBase**', type=SUPPORTED, accept_multiple_files=True, label_visibility='visible', on_change=new_doc)
    
    if st.session_state['new_doc'] and uploaded_files:
        for file in uploaded_files:
            file_name: str = file.name
            file_data: bytes = file.read()
            if file_name.lower().endswith('.zip'):
                unsupported = []
                try:
                    with zipfile.ZipFile(io.BytesIO(file_data)) as zip_folder:
                        for info in zip_folder.infolist():
                            if info.is_dir():
                                continue
                            member = info.filename
                            ext = member.rsplit('.', 1)[-1].lower()
                            if ext in SUPPORTED:
                                data = zip_folder.read(member)
                                safe_name = member.split('/')[-1]
                                success, error = process_file_via_api(safe_name, data, database_name)
                                if not success and error:
                                    st.session_state['processing_warnings'].append(error)
                            else:
                                unsupported.append(member)
                    if unsupported:
                        st.session_state['processing_warnings'].append('The following items in the ZIP were skipped because they are either folders or unsupported file types:\n\n' + '\n'.join((f'• {u}' for u in unsupported)) + '\n\nPlease include only PDF, DOCX, XLSX, TXT, or PPTX files.')
                except Exception as e:
                    st.session_state['processing_warnings'].append(f'The uploaded ZIP is corrupted or not a valid archive. Please upload a proper .zip containing only supported files.{e}')
            else:
                ext = file_name.rsplit('.', 1)[-1].lower()
                if ext in SUPPORTED:
                    success, error = process_file_via_api(file_name, file_data, database_name)
                    if not success and error:
                        st.session_state['processing_warnings'].append(error)
                else:
                    st.session_state['processing_warnings'].append(f'File `{file_name}` is not a supported type.')
        st.session_state['new_doc'] = False
        st.rerun()
