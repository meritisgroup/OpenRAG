from typing import List, Optional, Type
from database.rag_classes import Base, Chunk
from .database_manager import DatabaseManager
from .vector_store_manager import VectorStoreManager

class CoordinatedDataManager:

    def __init__(self, dbs_name: List[str], storage_path: str, data_folders_name: List[str], config_server: dict, agent=None):
        self.config_server = config_server
        self.storage_path = storage_path
        self.db_manager = DatabaseManager(storage_path=storage_path, config_server=config_server)
        self.vs_manager = VectorStoreManager(config_server=config_server, agent=agent)
        self.data_folders_name = []
        for i in range(len(dbs_name)):
            if isinstance(config_server.get('embedding_model'), list):
                embedding_model = config_server['embedding_model'][i]
            else:
                embedding_model = config_server.get('embedding_model', '')
            self.db_manager.add_database(db_name=dbs_name[i], data_folder_name=data_folders_name[i], storage_data_path=config_server.get('storage_data_path', './data/databases'))
            self.vs_manager.add_vectorstore(vb_name=dbs_name[i], embedding_model=embedding_model)

    def add_database(self, db_name: str, data_folder_name: str, embedding_model: str=None) -> None:
        storage_data_path = self.config_server.get('storage_data_path', './data/databases')
        self.db_manager.add_database(db_name, data_folder_name, storage_data_path)
        if embedding_model:
            self.vs_manager.add_vectorstore(db_name, embedding_model)

    def get_database(self, db_name: str):
        return self.db_manager.get_database(db_name)

    def set_database(self, db, db_name: str) -> None:
        self.db_manager.set_database(db, db_name)

    def get_instance_by_title(self, table_class: Type[Base], title: str, db_name: str=None):
        return self.db_manager.get_instance_by_title(table_class, title, db_name)

    def get_dbs_name(self) -> List[str]:
        return self.db_manager.get_dbs_name()

    def find_db_name_path(self, path: str) -> Optional[str]:
        return self.db_manager.find_db_name_path(path)

    def add_table(self, table_class: Type[Base], db_name: str=None, path: str=None):
        return self.db_manager.add_table(table_class, db_name, path)

    def add_instance(self, instance: Base, db_name: str=None, path: str=None) -> None:
        self.db_manager.add_instance(instance, db_name, path)

    def update_instance(self, db_name: str=None):
        self.db_manager.update_instance(db_name)

    def clean_database(self, db_name: str=None) -> None:
        self.db_manager.clean_database(db_name)

    def query(self, table_class: Type[Base], db_name: str=None, path_docs: str=None):
        return self.db_manager.query(table_class, db_name, path_docs)

    def query_filter(self, table_class: Type[Base], filter, db_name: str=None, path_docs: str=None):
        return self.db_manager.query_filter(table_class, filter, db_name, path_docs)

    def get_list_path_documents(self, db_name: str=None):
        return self.db_manager.get_list_path_documents(db_name)

    def create_merged_entities_table(self, db_name: str=None, overall: bool=True, path: str=None):
        self.db_manager.create_merged_entities_table(db_name, overall, path, None)

    def create_engine_connection(self):
        self.db_manager.create_engine_connections()

    def remove_engine_connection(self):
        self.db_manager.remove_engine_connections()

    def add_vectorbase(self, vb_name: str, embedding_model: str) -> None:
        self.vs_manager.add_vectorstore(vb_name, embedding_model)

    def set_vectorbase(self, vb_name: str, vb) -> None:
        self.vs_manager.set_vectorstore(vb_name, vb)

    def get_vectorbase(self, vb_name: str):
        return self.vs_manager.get_vectorstore(vb_name)

    def find_vb_name(self, path_docs: List[str]) -> Optional[str]:
        return self.vs_manager.find_vb_name(path_docs)

    def delete_collection(self, vb_name: str=None, name: str=None) -> None:
        self.vs_manager.delete_collection(vb_name, name)

    def create_collection(self, vb_name: str=None, name: str=None, add_fields: list=None) -> None:
        self.vs_manager.create_collection(vb_name, name, add_fields)

    def get_nb_token_embeddings(self, vb_name: str=None) -> int:
        return self.vs_manager.get_nb_token_embeddings(vb_name)

    def add_str_elements(self, chunks: List[Chunk], path_docs: List[str]=None, display_message: bool=True, collection_name: str=None, vb_name: str=None) -> int:
        return self.vs_manager.add_str_elements(chunks, path_docs, display_message, collection_name, vb_name)

    def add_str_batch_elements(self, chunks: List[Chunk], path_docs: List[str]=None, display_message: bool=True, collection_name: str=None, vb_name: str=None) -> int:
        return self.vs_manager.add_str_batch_elements(chunks, path_docs, display_message, collection_name, vb_name)

    def k_search(self, queries: List[str], k: int, vb_name: str=None, output_fields: List[str]=None, type_output=None, collection_name: str=None):
        return self.vs_manager.k_search(queries=queries, k=k, vb_name=vb_name, output_fields=output_fields, type_output=type_output, collection_name=collection_name)

def get_management_data(dbs_name: List[str], storage_path: str, data_folders_name: List[str], config_server: dict, agent=None) -> CoordinatedDataManager:
    return CoordinatedDataManager(dbs_name=dbs_name, storage_path=storage_path, data_folders_name=data_folders_name, config_server=config_server, agent=agent)