from sqlalchemy import inspect, create_engine, delete, MetaData
from sqlalchemy.orm import Session, Query
import os

from .rag_classes import (
    Base,
    Entity,
    Relation,
    MergeEntityDocument,
    MergeEntityOverall,
)
from .clean import DescriptionClean, Agent
from .rag_classes import Document, Chunk, Tokens
from tqdm.auto import tqdm


def get_database(db_name, storage_path):

    os.makedirs(storage_path, exist_ok=True)

    db = DataBase(db_name=db_name, path=storage_path)
    db.add_table(Document)

    return db


def get_graph_database(db_name, storage_path):
    db = DataBase(db_name=db_name, path=storage_path)
    db.add_table(Document)
    db.add_table(Chunk)
    db.add_table(Entity)
    db.add_table(Relation)
    db.add_table(Tokens)
    return db


class DataBase:

    def __init__(self, db_name: str, path: str):
        """
        Create a data base thanks to sqlalchemy
        """
        if len(db_name) <= 3 or db_name[-3:] != ".db":
            db_name += ".db"

        if path[-1] != "/":
            path += "/"

        self.name = db_name
        self.engine = create_engine(f"sqlite:///{path + db_name}")
        self.session = Session(bind=self.engine)

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

    def query(self, table_class: Base) -> Query:
        """
        Redefinition of sqlalchemy query operation to have easy access
        """
        return self.session.query(table_class)

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
