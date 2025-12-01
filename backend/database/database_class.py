from sqlalchemy import (
    inspect,
    create_engine,
    delete,
    MetaData,
    Column,
    String,
    Table,
    JSON,
)
from sqlalchemy.orm import Session, Query
from ..utils.factory_vectorbase import get_vectorbase
from .utils import get_list_path_documents
import os
from pathlib import Path
from sqlalchemy.orm import DeclarativeMeta
from typing import Type
from .rag_classes import Chunk

from .rag_classes import (
    Chunk,
    Base,
    Entity,
    Relation,
    MergeEntityDocument,
    MergeEntityOverall,
)
from .clean import DescriptionClean, Agent
from .rag_classes import Document, Chunk, Tokens
from tqdm.auto import tqdm
from typing import Union


def get_management_data(
    dbs_name, storage_path, data_folders_name, config_server: dict, agent: Agent = None
):
    db = Merger_Database_Vectorbase(dbs_name=dbs_name,
                                    storages_path=storage_path,
                                    agent=agent,
                                    config_server=config_server,
                                    storage_path=storage_path)
    print("dbs_name", dbs_name)
    print(config_server["embedding_model"])
    for i in range(len(dbs_name)):
        if type(config_server["embedding_model"])==list:
            db.add_database(db_name=dbs_name[i],
                            data_folder_name=data_folders_name[i],
                            embedding_model=config_server["embedding_model"][i])
        else:
            db.add_database(db_name=dbs_name[i],
                            data_folder_name=data_folders_name[i],
                            embedding_model=config_server["embedding_model"])

    return db


class Merger_Database_Vectorbase:

    def __init__(self, dbs_name, storages_path, agent, config_server, storage_path):

        self.databases = {}
        self.vectorbases = {}
        self.storage_path = storage_path
        self.data_folders_name = []
        self.agent = agent
        self.config_server = config_server
        self.auto_sharding = False

    def find_db_name_path(self, path: str) -> str:
        path = str(path)
        base_dir = Path(path).resolve()
        for db_name in self.databases.keys():
            candidate = Path(self.databases[db_name]["path"]).resolve()
            try:
                candidate.relative_to(base_dir)
                return db_name
            except ValueError:
                None

    def create_engine_connection(self):
        for db_name in self.databases.keys():
            self.databases[db_name]["database"].create_engine_connection()

    def remove_engine_connection(self):
        for db_name in self.databases.keys():
            self.databases[db_name]["database"].remove_engine_connection()

    def add_database(self, db_name, data_folder_name, embedding_model) -> None:
        save_path = os.path.join(self.storage_path, db_name)
        data_path = os.path.join(self.config_server["storage_data_path"],
                                 data_folder_name)

        self.databases[db_name] = {}
        self.databases[db_name]["path"] = data_path
        self.databases[db_name]["database"] = DataBase(db_name=db_name,
                                                       path=self.storage_path,
                                                       path_data=data_path)
        self.databases[db_name]["database"].add_table(Document)

        if not self.auto_sharding:
            self.add_vectorbase(vb_name=db_name,
                                embedding_model=embedding_model)

    def set_database(self, db, db_name: str):
        self.databases[db_name]["database"] = db

    def get_database(self, db_name: str):
        return self.databases[db_name]["database"]

    def get_dbs_name(self):
        return list(self.databases.keys())

    def add_table(self, table_class: Base, db_name: str = None, path: str = None):
        if db_name is None and path is not None:
            db_name = self.find_db_name_path(path)
        if db_name is not None:
            return self.databases[db_name]["database"].add_table(table_class)
        else:
            for db_name in self.databases.keys():
                self.databases[db_name]["database"].add_table(table_class)

    def add_instance(
        self, instance: Base, db_name: str = None, path: str = None
    ) -> None:
        if db_name is None:
            db_name = self.find_db_name_path(path)
        if db_name is not None:
            self.databases[db_name]["database"].add_instance(instance=instance)

    def update_instance(self, db_name):
        if db_name is not None:
            self.databases[db_name]["database"].update_instance()
        else:
            for db_name in self.databases.keys():
                self.databases[db_name]["database"].update_instance()

    def create_merged_entities_table(
        self, db_name: str = None, overall=True, path: str = None
    ):
        if db_name is None and path is not None:
            db_name = self.find_db_name_path(path)

        if db_name is not None:
            self.databases[db_name]["database"].create_merged_entities_table(
                agent=self.agent, overall=overall
            )
        else:
            for db_name in self.databases.keys():
                self.databases[db_name]["database"].create_merged_entities_table(
                    agent=self.agent, overall=overall
                )

    def clean_database(self, db_name: str = None) -> None:
        if db_name is not None:
            self.databases[db_name]["database"].clean_database()
        else:
            for db_name in self.databases.keys():
                self.databases[db_name]["database"].clean_database()

    def query(self, table_class: Base, db_name: str = None, path_docs: str = None):
        if db_name is None and path_docs is not None:
            db_name = self.find_db_name_path(path_docs)

        if db_name is not None:
            return self.databases[db_name]["database"].query(table_class)
        else:
            results = []
            for db_name in self.databases.keys():
                results += self.databases[db_name]["database"].query(table_class)
            return results

    def query_filter(
        self, table_class: Base, filter, db_name: str = None, path_docs: str = None
    ):
        if db_name is None and path_docs is not None:
            db_name = self.find_db_name_path(path_docs)

        if db_name is not None:
            return self.databases[db_name]["database"].query_filter(
                table_class=table_class, filter=filter
            )
        else:
            results = []
            for db_name in self.databases.keys():
                results += self.databases[db_name]["database"].query_filter(
                    table_class=table_class, filter=filter
                )
        return results

    def get_list_path_documents(self, db_name: str = None):
        if db_name is not None:
            return self.databases[db_name]["database"].get_list_path_documents()
        else:
            results = []
            for db_name in self.databases.keys():
                results += self.databases[db_name]["database"].get_list_path_documents()
            return results

    def add_vectorbase(self, vb_name, embedding_model) -> None:
        self.vectorbases[vb_name] = {}
        self.vectorbases[vb_name]["vectorbase"] = get_vectorbase(vb_name=vb_name,
                                                                 config_server=self.config_server,
                                                                 agent=self.agent,
                                                                 embedding_model=embedding_model)

    def set_vectorbase(self, vb_name, vb):
        self.vectorbases[vb_name]["vectorbase"] = vb

    def get_vectorbase(self, vb_name):
        return self.vectorbases[vb_name]["vectorbase"]

    def find_vb_name(self, path_docs):
        return self.find_db_name_path(path=path_docs[0])

    def delete_collection(self, vb_name: str = None, name: str = None) -> None:
        if vb_name is not None:
            if name is not None:
                name_delete = vb_name + "_" + name
            else:
                name_delete = None
            self.vectorbases[vb_name]["vectorbase"].delete_collection(vb_name=name_delete)
        else:
            for vb_name in self.vectorbases.keys():
                if name is not None:
                    name_delete = vb_name + "_" + name
                else:
                    name_delete = None
                self.vectorbases[vb_name]["vectorbase"].delete_collection(vb_name=name_delete)

    def create_collection(self, vb_name: str = None, name: str = None, add_fields=[]) -> None:
        if vb_name is not None:
            if name is not None:
                name_create = vb_name + "_" + name
            else:
                name_create = vb_name
            self.vectorbases[vb_name]["vectorbase"].create_collection(
                name=name_create, add_fields=add_fields
            )
        else:
            for vb_name in self.vectorbases.keys():
                if name is not None:
                    name_create = vb_name + "_" + name
                else:
                    name_create = vb_name

                self.vectorbases[vb_name]["vectorbase"].create_collection(
                    name=name_create, add_fields=add_fields
                )

    def get_nb_token_embeddings(self, vb_name: str = None) -> None:
        if vb_name is not None:
            return self.vectorbases[vb_name]["vectorbase"].get_nb_token_embeddings()
        else:
            nb_token_embeddings = 0
            for vb_name in self.vectorbases.keys():
                nb_token_embeddings += self.vectorbases[vb_name][
                    "vectorbase"
                ].get_nb_token_embeddings()
            return nb_token_embeddings

    def add_str_elements(
        self,
        chunks=list[Chunk],
        path_docs: list[str] = None,
        display_message: bool = True,
        collection_name=None,
        vb_name: str = None,
    ) -> None:
        if vb_name is None:
            vb_name = self.find_vb_name(path_docs)
        elif collection_name is not None and vb_name is not None:
            collection_name = vb_name + "_" + collection_name

        if vb_name is not None:
            return self.vectorbases[vb_name]["vectorbase"].add_str_elements(
                chunks=chunks,
                display_message=display_message,
                collection_name=collection_name,
            )

    def add_str_batch_elements(
        self,
        chunks: list[Chunk],
        path_docs: list[str] = None,
        display_message: bool = True,
        collection_name=None,
        vb_name: str = None,
    ) -> None:

        if vb_name is None:
            vb_name = self.find_vb_name(path_docs)
        elif collection_name is not None and vb_name is not None:
            collection_name = vb_name + "_" + collection_name

        if vb_name is not None:
            return self.vectorbases[vb_name]["vectorbase"].add_str_batch_elements(
                chunks=chunks,
                display_message=display_message,
                collection_name=collection_name,
            )

    def k_search(
        self,
        queries: Union[str, list[str]],
        k: int,
        output_fields: list[str] = ["text"],
        filters: dict = None,
        collection_name=None,
        vb_name: str = None,
        type_output = Chunk
    ):
        if vb_name is not None:
            if collection_name is not None:
                collection_name = vb_name + "_" + collection_name
            return self.vectorbases[vb_name]["vectorbase"].k_search(
                queries=queries,
                k=k,
                output_fields=output_fields,
                filters=filters,
                collection_name=collection_name,
                type_output=type_output
            )
        else:
            n = len(self.vectorbases)
            k_per_vectorbase = k // n
            remainder = k % n

            k_per_db_list = [k_per_vectorbase] * n
            for i in range(remainder):
                k_per_db_list[i] += 1

            chunks = []
            for i in range(len(self.vectorbases)):
                vb_name = list(self.vectorbases.keys())[i]
                if collection_name is not None:
                    collection_name_search = vb_name + "_" + collection_name
                else:
                    collection_name_search = collection_name

                result = self.vectorbases[vb_name]["vectorbase"].k_search(
                    queries=queries,
                    k=k_per_db_list[i],
                    output_fields=output_fields,
                    collection_name=collection_name_search,
                    type_output=type_output
                )
                if len(chunks) == 0:
                    chunks = result
                else:
                    for j in range(len(result)):
                        chunks[j] += result[j]
            return chunks


class DataBase:

    def __init__(self, db_name: str, path: str, path_data: str):
        """
        Create a data base thanks to sqlalchemy
        """
        if len(db_name) <= 3 or db_name[-3:] != ".db":
            db_name += ".db"

        if path[-1] != "/":
            path += "/"

        self.path = path
        self.path_data = path_data

        self.name = db_name
        self.create_engine_connection()

    def create_engine_connection(self):
        self.engine = create_engine(f"sqlite:///{self.path + self.name}")
        self.session = Session(bind=self.engine)

    def remove_engine_connection(self):
        self.engine = None
        self.session = None

    def add_table(self, table_class: Base) -> None:
        """
        Add the table associated to table_class which has to be the class itself not an element from the class
        """
        inspector = inspect(self.engine)

        if table_class.__tablename__ not in inspector.get_table_names():
            table_class.__table__.create(self.engine, checkfirst=True)

        else:
            print(f'The table "{table_class.__tablename__}" already exists.')

    def remove_table(self, table_class: Base) -> None:
        """
        Remove the table associated to table_class which has to be the class itself not an element from the class
        """
        inspector = inspect(self.engine)

        if table_class.__tablename__ in inspector.get_table_names():
            table_class.__table__.drop(self.engine, checkfirst=True)

        else:
            print(
                f'Impossible to drop "{table_class.__tablename__}" because it doesn\'t exist in data base.'
            )

    def add_instance(self, instance: Base) -> None:
        """
        Add an instance in the right table
        """
        try:
            self.session.add(instance)
            self.session.commit()

        except Exception as e:
            self.session.rollback()
            print(f'An error "{e}" occured when trying to add the instance')

    def update_instance(self) -> None:
        """
        Commit changes to an instance already attached to the session.
        """
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f'An error "{e}" occurred when trying to update the instance')

    def delete_instance(self, instance: Base) -> None:
        """
        Delete an instance from the database.
        """
        try:
            self.session.delete(instance)
            self.session.commit()

        except Exception as e:
            self.session.rollback()
            print(f'An error "{e}" occurred when trying to delete the instance.')

    def get_instance_by_title(self, table_class: Base, title: str):
        """
        Retrieve an instance from the given table class by its title.
        """
        try:
            return self.session.query(table_class).filter_by(title=title).first()
        except Exception as e:
            print(
                f'An error "{e}" occurred when trying to retrieve the instance by title'
            )

        return None

    def query(self, table_class: Base):
        """
        Redefinition of sqlalchemy query operation to have easy access
        """
        return self.session.query(table_class).all()

    def query_filter(self, table_class: Base, filter):
        return self.session.query(table_class).filter(filter).all()

    def create_merged_entities_table(self, agent: Agent, overall=True) -> None:
        """
        Merges entities descriptions of the same entity. If overall == False, it only merges for same document.
        """
        entity_names = set([res[0] for res in self.session.query(Entity.name).all()])

        if overall:
            self.add_table(MergeEntityOverall)

            already_in_entities = [
                res[0] for res in self.session.query(MergeEntityOverall.name).all()
            ]
            entities_to_process = [
                entity_name
                for entity_name in entity_names
                if entity_name not in already_in_entities
            ]

            with tqdm(entities_to_process) as progress_bar:
                for k, entity_name in enumerate(progress_bar):
                    progress_bar.set_description(
                        f"Merging Multiples Occurences - {entity_name} "
                    )

                    entities = (
                        self.session.query(Entity)
                        .filter(Entity.name == entity_name)
                        .all()
                    )

                    descriptions = [entity.description for entity in entities]
                    chunk_ids = [entity.chunk_id for entity in entities]
                    doc_names = [entity.doc_name for entity in entities]
                    kind = entities[0].kind
                    description = DescriptionClean(
                        descriptions=descriptions
                    ).clean_description(agent=agent)

                    new_entity = MergeEntityOverall(
                        name=entity_name,
                        kind=kind,
                        description=description,
                        chunk_ids=chunk_ids,
                        doc_names=doc_names,
                        degree=len(entities),
                    )

                    self.session.add(new_entity)
                    self.session.commit()

                    if k == len(progress_bar) - 1:
                        progress_bar.set_description(
                            "Merging multiples occurences - ✅"
                        )

        else:
            self.add_table(MergeEntityDocument)

            with tqdm(entity_names) as progress_bar:

                for k, entity_name in enumerate(progress_bar):

                    progress_bar.set_description(
                        f"Merging Multiples Occurences - {entity_name} "
                    )

                    entities = (
                        self.session.query(Entity)
                        .filter(Entity.name == entity_name)
                        .all()
                    )

                    doc_names = [entity.doc_name for entity in entities]

                    for doc_name in doc_names:

                        already_in_entities = [
                            res[0]
                            for res in self.session.query(MergeEntityDocument.name)
                            .filter(MergeEntityDocument.doc_name == doc_name)
                            .all()
                        ]

                        if entity_name not in already_in_entities:

                            filtered_entities = [
                                entity
                                for entity in entities
                                if entity.doc_name == doc_name
                            ]

                            descriptions = [
                                entity.description for entity in filtered_entities
                            ]
                            chunk_ids = [
                                entity.chunk_id for entity in filtered_entities
                            ]

                            kind = filtered_entities[0].kind

                            description = DescriptionClean(
                                descriptions=descriptions
                            ).clean_description(agent=agent)

                            new_entity = MergeEntityDocument(
                                name=entity_name,
                                kind=kind,
                                description=description,
                                chunk_ids=chunk_ids,
                                doc_name=doc_name,
                                degree=len(filtered_entities),
                            )

                            self.session.add(new_entity)
                            self.session.commit()

                    if k == len(progress_bar) - 1:
                        progress_bar.set_description(
                            "Merging multiples occurences - ✅"
                        )

    def clean_database(self) -> None:
        """
        Clean all data from all existing tables in the database while keeping the table structures intact.
        """
        try:
            meta = MetaData()
            meta.reflect(bind=self.engine)

            # 2. Open a transaction
            with self.engine.begin() as conn:
                # 3. Delete in reverse order to satisfy FK constraints
                for table in reversed(meta.sorted_tables):
                    conn.execute(delete(table))

            print(
                f"Successfully cleaned {len(meta.tables)} tables: "
                f"{', '.join(meta.tables)}"
            )

        except Exception as e:
            self.session.rollback()
            print(f'An error "{e}" occurred when trying to clean the database')
    def get_list_path_documents(self):
        return get_list_path_documents(path_data=self.path_data)


class ContextDatabase:

    def __init__(
        self, db_name: str = "context_rags.db", db_folder_path: str = "./storage"
    ):
        self.db_name = db_name
        self.metadata = MetaData()
        self.db_path = os.path.join(db_folder_path, db_name)
        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def create_rag_table(self, rag_name: str):
        """
        Dynamically create a table for a RAG.
        Each table has:
        - query (string, primary key)
        - context (JSON: list of chunks)
        """

        rag_table = Table(
            rag_name,
            self.metadata,
            Column("query", String, primary_key=True),
            Column("context", JSON),
            extend_existing=True,
        )
        self.metadata.create_all(self.engine)
        return rag_table

    def insert_results(self, rag_table, query: str, chunks: list[Chunk]):
        """Insert one row (query + list of chunk dicts) into a given RAG table."""
        dict_chunks = [chunk.to_dict() for chunk in chunks]
        with self.engine.begin() as conn:
            sel = rag_table.select().where(rag_table.c.query == query)
            result = conn.execute(sel).first()

            if result is None:
                ins = rag_table.insert().values(query=query, 
                                                context=dict_chunks)
                conn.execute(ins)
            else:
                print(
                    f"Query already exists in table '{rag_table.name}', skipping insert."
                )

    def complete_context_database(
        self, rag_name: str, queries: list[str], answers: list[dict]
    ):
        table = self.create_rag_table(rag_name)
        for query, answer in zip(queries, answers):
            self.insert_results(table, query, answer["context"])