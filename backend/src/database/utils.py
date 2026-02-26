import os
from pathlib import Path

def get_list_path_documents(path_data):
    docs_path = os.path.join(path_data, 'documents')
    if not os.path.exists(docs_path):
        docs_path = path_data
    all_files = os.listdir(docs_path)
    all_files = [os.path.join(docs_path, doc_name) for doc_name in all_files if doc_name != 'metadatas.json' and (not Path(os.path.join(docs_path, doc_name)).is_dir())]
    return all_files