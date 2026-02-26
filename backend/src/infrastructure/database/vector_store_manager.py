from typing import Optional, List
from pathlib import Path
from database.rag_classes import Chunk
from utils.factory_vectorbase import get_vectorbase

class VectorStoreManager:

    def __init__(self, config_server: dict, agent=None):
        self.vectorbases = {}
        self.config_server = config_server
        self.agent = agent

    def add_vectorstore(self, vb_name: str, embedding_model: str) -> None:
        self.vectorbases[vb_name] = {}
        self.vectorbases[vb_name]['vectorbase'] = get_vectorbase(vb_name=vb_name, config_server=self.config_server, agent=self.agent, embedding_model=embedding_model)

    def set_vectorstore(self, vb_name: str, vb) -> None:
        if vb_name not in self.vectorbases:
            self.vectorbases[vb_name] = {}
        self.vectorbases[vb_name]['vectorbase'] = vb

    def get_vectorstore(self, vb_name: str):
        if vb_name in self.vectorbases:
            return self.vectorbases[vb_name]['vectorbase']
        return None

    def get_vectorstores_names(self) -> List[str]:
        return list(self.vectorbases.keys())

    def find_vb_name(self, path_docs: List[str]) -> Optional[str]:
        if not path_docs or not self.vectorbases:
            return None
        first_doc_path = Path(path_docs[0])
        parent_folder = first_doc_path.parent.name
        if parent_folder in self.vectorbases:
            return parent_folder
        return list(self.vectorbases.keys())[0]

    def create_collection(self, vb_name: str=None, name: str=None, add_fields: list=None) -> None:
        if vb_name is not None:
            if name is not None:
                name_create = vb_name + '_' + name
            else:
                name_create = vb_name
            self.vectorbases[vb_name]['vectorbase'].create_collection(name=name_create, add_fields=add_fields or [])
        else:
            for vb_name in self.vectorbases.keys():
                if name is not None:
                    name_create = vb_name + '_' + name
                else:
                    name_create = vb_name
                self.vectorbases[vb_name]['vectorbase'].create_collection(name=name_create, add_fields=add_fields or [])

    def delete_collection(self, vb_name: str=None, name: str=None) -> None:
        if vb_name is not None:
            if name is not None:
                name_delete = vb_name + '_' + name
            else:
                name_delete = None
            self.vectorbases[vb_name]['vectorbase'].delete_collection(vb_name=name_delete)
        else:
            for vb_name in self.vectorbases.keys():
                if name is not None:
                    name_delete = vb_name + '_' + name
                else:
                    name_delete = None
                self.vectorbases[vb_name]['vectorbase'].delete_collection(vb_name=name_delete)

    def get_nb_token_embeddings(self, vb_name: str=None) -> int:
        if vb_name is not None:
            return self.vectorbases[vb_name]['vectorbase'].get_nb_token_embeddings()
        else:
            nb_token_embeddings = 0
            for vb_name in self.vectorbases.keys():
                nb_token_embeddings += self.vectorbases[vb_name]['vectorbase'].get_nb_token_embeddings()
            return nb_token_embeddings

    def add_str_elements(self, chunks: List[Chunk], path_docs: List[str]=None, display_message: bool=True, collection_name: str=None, vb_name: str=None) -> int:
        if vb_name is None:
            vb_name = self.find_vb_name(path_docs)
        elif collection_name is not None and vb_name is not None:
            collection_name = vb_name + '_' + collection_name
        if vb_name is not None:
            return self.vectorbases[vb_name]['vectorbase'].add_str_elements(chunks=chunks, display_message=display_message, collection_name=collection_name)
        return 0

    def add_str_batch_elements(self, chunks: List[Chunk], path_docs: List[str]=None, display_message: bool=True, collection_name: str=None, vb_name: str=None) -> int:
        if vb_name is None:
            vb_name = self.find_vb_name(path_docs)
        elif collection_name is not None and vb_name is not None:
            collection_name = vb_name + '_' + collection_name
        if vb_name is not None:
            return self.vectorbases[vb_name]['vectorbase'].add_str_batch_elements(chunks=chunks, display_message=display_message, collection_name=collection_name)
        return 0

    def k_search(self, queries: List[str], k: int, vb_name: str=None, output_fields: List[str]=None, type_output='Chunk', collection_name: str=None):
        if type_output is None or type_output == 'Chunk':
            type_output = Chunk
        if vb_name is not None:
            if collection_name is not None:
                collection_name = vb_name + '_' + collection_name
            print(f'[DEBUG] k_search: vb_name={vb_name}, collection_name={collection_name}')
            return self.vectorbases[vb_name]['vectorbase'].k_search(queries=queries, k=k, output_fields=output_fields, type_output=type_output, collection_name=collection_name)
        else:
            results = []
            for vb_name in self.vectorbases.keys():
                collection_name_search = collection_name
                if collection_name is not None:
                    collection_name_search = vb_name + '_' + collection_name
                print(f'[DEBUG] k_search: vb_name={vb_name}, collection_name_search={collection_name_search}')
                results += self.vectorbases[vb_name]['vectorbase'].k_search(queries=queries, k=k, output_fields=output_fields, type_output=type_output, collection_name=collection_name_search)
            return results