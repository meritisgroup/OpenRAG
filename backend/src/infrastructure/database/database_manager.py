import os
from typing import Type, Optional, List
from pathlib import Path
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session, DeclarativeMeta
from utils.progress import tqdm
import numpy as np

from database.rag_classes import Base, Document, Entity, Relation, MergeEntityOverall, MergeEntityDocument
from database.clean import DescriptionClean, Agent

class DataBase:

    def __init__(self, db_name: str, path: str, path_data: str):
        self.db_name = db_name
        self.path = path
        self.path_data = path_data
        self.engine = None
        self.session = None

    def create_engine_connection(self):
        os.makedirs(self.path, exist_ok=True)
        db_path = os.path.join(self.path, self.db_name)
        self.engine = create_engine(f'sqlite:///{db_path}.db')

    def remove_engine_connection(self):
        if self.engine:
            self.engine.dispose()
            self.engine = None

    def add_table(self, table_class: Type[Base]):
        if self.engine is None:
            self.create_engine_connection()
        table_class.metadata.create_all(self.engine)

    def add_instance(self, instance: Base):
        if self.session is None:
            self.session = Session(bind=self.engine)
        self.session.add(instance)
        self.session.commit()

    def update_instance(self):
        if self.session:
            self.session.commit()

    def query(self, table_class: Type[Base]):
        if self.session is None:
            self.session = Session(bind=self.engine)
        return self.session.query(table_class)

    def query_filter(self, table_class: Type[Base], filter):
        return self.query(table_class).filter(filter)

    def clean_database(self):
        if self.engine is None:
            return
        with self.engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(table.delete())

    def get_instance_by_title(self, table_class: Type[Base], title: str):
        try:
            return self.query(table_class).filter_by(title=title).first()
        except Exception as e:
            print(f'An error "{e}" occurred when trying to retrieve the instance by title')
        return None

    def get_list_path_documents(self) -> List[str]:
        docs_path = os.path.join(self.path_data, 'documents')
        if not os.path.exists(docs_path):
            docs_path = self.path_data
        all_files = os.listdir(docs_path)
        all_files = [os.path.join(docs_path, doc_name) for doc_name in all_files if doc_name != 'metadatas.json' and (not Path(os.path.join(docs_path, doc_name)).is_dir())]
        return all_files

    def create_merged_entities_table(self, agent: Agent=None, overall: bool=True):
        print(f'[DEBUG] DataBase.create_merged_entities_table: START')
        entity_names = set([res[0] for res in self.query(Entity.name)])
        print(f'[DEBUG] create_merged_entities_table: Found {len(entity_names)} unique entity names in Entity table')

        if overall:
            self.add_table(MergeEntityOverall)
            already_in_entities = [res[0] for res in self.query(MergeEntityOverall.name)]
            print(f'[DEBUG] create_merged_entities_table: Already merged {len(already_in_entities)} entities in MergeEntityOverall')
            entities_to_process = [entity_name for entity_name in entity_names if entity_name not in already_in_entities]
            print(f'[DEBUG] create_merged_entities_table: Entities to process: {len(entities_to_process)}')
            if len(entities_to_process) > 0:
                with tqdm(entities_to_process) as progress_bar:
                    for (k, entity_name) in enumerate(progress_bar):
                        progress_bar.set_description(f'Merging Multiples Occurences - {entity_name} ')
                        entities = self.query_filter(Entity, Entity.name == entity_name).all()
                        descriptions = [entity.description for entity in entities]
                        chunk_ids = [entity.chunk_id for entity in entities]
                        doc_names = [entity.doc_name for entity in entities]
                        kind = entities[0].kind
                        description = DescriptionClean(descriptions=descriptions).clean_description(agent=agent)
                        new_entity = MergeEntityOverall(name=entity_name, kind=kind, description=description, chunk_ids=chunk_ids, doc_names=doc_names, degree=len(entities))
                        self.add_instance(new_entity)
                        if k == len(progress_bar) - 1:
                            progress_bar.set_description('Merging multiples occurences - ✅')
            final_count = len([res[0] for res in self.query(MergeEntityOverall.name)])
            print(f'[DEBUG] create_merged_entities_table: Final count in MergeEntityOverall: {final_count}')

class DatabaseManager:

    def __init__(self, storage_path: str, config_server: dict=None):
        self.databases = {}
        self.storage_path = storage_path
        self.config_server = config_server

    def add_database(self, db_name: str, data_folder_name: str, storage_data_path: str) -> DataBase:
        data_path = os.path.join(storage_data_path, data_folder_name)
        self.databases[db_name] = {}
        self.databases[db_name]['path'] = data_path
        self.databases[db_name]['database'] = DataBase(db_name=db_name, path=self.storage_path, path_data=data_path)
        self.databases[db_name]['database'].add_table(Document)
        return self.databases[db_name]['database']

    def get_database(self, db_name: str) -> Optional[DataBase]:
        if db_name in self.databases:
            return self.databases[db_name]['database']
        return None

    def set_database(self, db: DataBase, db_name: str) -> None:
        if db_name not in self.databases:
            self.databases[db_name] = {}
        self.databases[db_name]['database'] = db

    def get_instance_by_title(self, table_class: Type[Base], title: str, db_name: str=None):
        if db_name is not None:
            db = self.get_database(db_name)
            if db:
                return db.get_instance_by_title(table_class, title)
        else:
            for db_name in self.databases.keys():
                db = self.databases[db_name]['database']
                result = db.get_instance_by_title(table_class, title)
                if result is not None:
                    return result
        return None

    def get_dbs_name(self) -> List[str]:
        return list(self.databases.keys())

    def find_db_name_path(self, path: str) -> Optional[str]:
        path = str(path)
        base_dir = Path(path).resolve()
        for db_name in self.databases.keys():
            candidate = Path(self.databases[db_name]['path']).resolve()
            try:
                candidate.relative_to(base_dir)
                return db_name
            except ValueError:
                pass
        return None

    def create_engine_connections(self) -> None:
        for db_name in self.databases.keys():
            self.databases[db_name]['database'].create_engine_connection()

    def remove_engine_connections(self) -> None:
        for db_name in self.databases.keys():
            self.databases[db_name]['database'].remove_engine_connection()

    def add_table(self, table_class: Type[Base], db_name: str=None, path: str=None) -> None:
        if db_name is None and path is not None:
            db_name = self.find_db_name_path(path)
        if db_name is not None:
            self.databases[db_name]['database'].add_table(table_class)
        else:
            for db_name in self.databases.keys():
                self.databases[db_name]['database'].add_table(table_class)

    def add_instance(self, instance: Base, db_name: str=None, path: str=None) -> None:
        if db_name is None:
            db_name = self.find_db_name_path(path)
        if db_name is not None:
            self.databases[db_name]['database'].add_instance(instance=instance)

    def update_instance(self, db_name: str=None) -> None:
        if db_name is not None:
            self.databases[db_name]['database'].update_instance()
        else:
            for db_name in self.databases.keys():
                self.databases[db_name]['database'].update_instance()

    def clean_database(self, db_name: str=None) -> None:
        if db_name is not None:
            self.databases[db_name]['database'].clean_database()
        else:
            for db_name in self.databases.keys():
                self.databases[db_name]['database'].clean_database()

    def query(self, table_class: Type[Base], db_name: str=None, path_docs: str=None):
        if db_name is None and path_docs is not None:
            db_name = self.find_db_name_path(path_docs)
        if db_name is not None:
            return self.databases[db_name]['database'].query(table_class)
        else:
            results = []
            for db_name in self.databases.keys():
                results += self.databases[db_name]['database'].query(table_class)
            return results

    def query_filter(self, table_class: Type[Base], filter, db_name: str=None, path_docs: str=None):
        if db_name is None and path_docs is not None:
            db_name = self.find_db_name_path(path_docs)
        if db_name is not None:
            return self.databases[db_name]['database'].query_filter(table_class=table_class, filter=filter)
        else:
            results = []
            for db_name in self.databases.keys():
                results += self.databases[db_name]['database'].query_filter(table_class=table_class, filter=filter)
            return results

    def get_list_path_documents(self, db_name: str=None) -> List[str]:
        if db_name is not None:
            return self.databases[db_name]['database'].get_list_path_documents()
        else:
            results = []
            for db_name in self.databases.keys():
                results += self.databases[db_name]['database'].get_list_path_documents()
            return results

    def create_merged_entities_table(self, db_name: str=None, overall: bool=True, path: str=None, agent=None) -> None:
        if db_name is None and path is not None:
            db_name = self.find_db_name_path(path)
        if db_name is not None:
            self.databases[db_name]['database'].create_merged_entities_table(agent=agent, overall=overall)
        else:
            for db_name in self.databases.keys():
                self.databases[db_name]['database'].create_merged_entities_table(agent=agent, overall=overall)
