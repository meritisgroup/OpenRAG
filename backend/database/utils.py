import os
from pathlib import Path


def get_list_path_documents(path_data):
        all_files = os.listdir(path_data)
        all_files = [
            os.path.join(path_data, doc_name)
            for doc_name in all_files
            if doc_name != "metadatas.json" and not Path(os.path.join(path_data, doc_name)).is_dir()
        ]
        all_docs = [f for f in all_files]
        return all_docs