from ...database.rag_classes import (
    Relation,
    Community,
    MergeEntityOverall,
)
from ...database.database_class import DataBase
from .extract_entities import extract_entities_relations
from ...database.rag_classes import DocumentText
from .graph_creation import Graph
from .community_description import CommunityDescription
from ...utils.splitter import get_splitter
from ...utils.agent import Agent
from ...database.rag_classes import Document, Tokens

from tqdm import tqdm
import numpy as np
from sqlalchemy.orm import Session
import os

from ...utils.factory_name_dataser_vectorbase import get_name
from ...utils.progress import ProgressBar


class GraphRagIndexation:
    def __init__(
        self,
        data_path: str,
        storage_path: str,
        agent: Agent,
        db: DataBase,
        vb,
        type_text_splitter: str,
        embedding_model: str,
        language: str = "EN",
    ):
        self.db = db
        self.vb = vb
        self.vb_name = self.db.name
        self.agent = agent
        self.language = language

        if data_path[-1] != "/":
            data_path += "/"

        self.splitter = get_splitter(
            type_text_splitter=type_text_splitter,
            embedding_model=embedding_model,
            agent=self.agent,
        )
        self.data_path = data_path
        self.storage_path = storage_path

    def check_extraction_necessity(self):
        need_extraction = False

        db_name = self.db.name
        file_path = os.path.join(self.storage_path, db_name)

        documents_to_process = os.listdir(self.data_path)

        if os.path.isfile(file_path):
            with Session(self.db.engine) as session:
                documents = session.query(Document).all()
                names = [doc.name for doc in documents]

            difference = [doc for doc in documents_to_process if doc not in names]

            if difference != []:
                need_extraction = True
        else:
            need_extraction = True
        return need_extraction

    def extract_entities(self, chunk_size: int = 500, overlap: bool = True):
        """
        Extract entities and relations from texts located in the folder self.data_path and save all the results in the data base self.db
        """
        documents_nb_input_tokens = 0
        documents_nb_output_tokens = 0

        progress_bar = ProgressBar(total=len(os.listdir(self.data_path)))
        for k, doc_name in enumerate(os.listdir(self.data_path)):
            progress_bar.update(k-1, text="Entities extraction - Processing file {}".format(doc_name))

            document = DocumentText(
                    path=f"{self.data_path}/{doc_name}", splitter=self.splitter
                )
            chunks = document.chunks(chunk_size=chunk_size, chunk_overlap=overlap)

            for chunk in chunks:
                self.db.add_instance(chunk)

            entities, relations, input_tokens, output_tokens = (
                    extract_entities_relations(
                        agent=self.agent,
                        chunks=chunks,
                        doc_name=doc_name,
                        language=self.language,
                    )
                )
            documents_nb_input_tokens += np.sum(input_tokens)
            documents_nb_output_tokens += np.sum(output_tokens)

            document.input_tokens = np.sum(input_tokens)
            document.output_tokens = np.sum(output_tokens)

            for instance in entities:
                self.db.add_instance(instance)
            for instance in relations:
                self.db.add_instance(instance)

            document_base = document.convert_in_base()
            self.db.add_instance(document_base)

        progress_bar.update(k, text="Entities extraction - Processing file {}".format(doc_name))
        
        documents_tokens = Tokens(
            title="documents",
            embedding_tokens=0,
            input_tokens=int(documents_nb_input_tokens),
            output_tokens=int(documents_nb_output_tokens),
        )
        self.db.add_instance(documents_tokens)

    def clean_entities(self):
        """
        If an entity has been retrieved multiple times, it merges all the description in one.
        """
        self.db.create_merged_entities_table(self.agent)

    def create_graph(self):
        """
        Create and clusterize the graph of merged entities thanks to relations found in extract_entities process
        """
        entities = self.db.query(MergeEntityOverall).all()
        relations = self.db.query(Relation).all()

        g = Graph(entities, relations)

        g.create_communities()
        g.plot_graph(storage_path=self.storage_path, graph_name="graph_rag")

        return g

    def describe_communities(self, graph: Graph):
        """
        Generate a title and a description for each cluster of the provided graph and save it into the data base
        """
        self.db.add_table(Community)

        cd = CommunityDescription(agent=self.agent, graph=graph, database=self.db)

        cd.process_communities(language=self.language)

    def save_vb_local(self):
        """
        Save entities' names in a vector base
        """
        entities_nb_tokens = 0

        self.vb.create_collection(name=self.db.name[:-3] + "_local_search")
        entities = [res[0] for res in self.db.query(MergeEntityOverall.name).all()]

        with tqdm(
            range(1 + len(entities) // 10), desc="Embedding for local search"
        ) as progress_bar:

            for k in progress_bar:
                truncated_entities = entities[10 * k : min(10 * (k + 1), len(entities))]

                if truncated_entities != []:
                    nb_tokens = 0
                    taille_batch = 500
                    for j in range(0, len(truncated_entities), taille_batch):
                        nb_tokens = np.sum(self.vb.add_str_batch_elements(
                                            collection_name=self.db.name[:-3] + "_local_search",
                                            elements=truncated_entities[j:j + taille_batch],
                                            display_message=False
                                    ))

                        entities_nb_tokens += np.sum(nb_tokens)
                if k == len(entities) // 10:
                    progress_bar.set_description("Embedding for local search - ✅")

        entities_tokens = Tokens(
            title="entities",
            embedding_tokens=int(entities_nb_tokens),
            input_tokens=0,
            output_tokens=0,
        )
        self.db.add_instance(entities_tokens)

    def save_vb_global(self):
        """
        Save ecollection titles in a vector base
        """
        communities_embedding_tokens = 0
        community_instance: Tokens = self.db.get_instance_by_title(
            Tokens, "communities"
        )

        self.vb.create_collection(name=self.db.name[:-3] + "_global_search")
        communities = [res.title for res in self.db.query(Community).all()]

        with tqdm(
            range(1 + len(communities) // 10), desc="Embedding for global search"
        ) as progress_bar:
            for k in progress_bar:
                truncated_communities = communities[
                    10 * k : min(10 * (k + 1), len(communities))
                ]

                if truncated_communities != []:
                    nb_tokens = 0
                    taille_batch = 500
                    for j in range(0, len(truncated_communities), taille_batch):
                        nb_tokens = np.sum(self.vb.add_str_batch_elements(
                                            collection_name=self.db.name[:-3] + "_global_search",
                                            elements=truncated_communities[j:j + taille_batch],
                                            display_message=False
                                    ))

                        communities_embedding_tokens += np.sum(nb_tokens)

                if k == len(communities) // 10:
                    progress_bar.set_description("Embedding for global search - ✅")

        community_instance.embedding_tokens = communities_embedding_tokens
        self.db.update_instance()

    def run_pipeline(self, chunk_size: int = 500, overlap: bool = True):
        """
        Indexation phase for graph rag, from entities extraction to community description.
        """
        print("Checking extraction necessity")
        need_extraction = self.check_extraction_necessity()

        if need_extraction:
            progress_bar = ProgressBar(total=5)
            print("Starting extration\n")
            progress_bar.update(0, text="Extraction of entities and relations\n")
            self.extract_entities(chunk_size=chunk_size, overlap=overlap)
            print("Extration done - ✅\n")
            progress_bar.update(1, text="Merging muliples entities of entities")
            self.clean_entities()
            print("Creating the graph and the communities")
            progress_bar.update(2, text="Creation of the graph")
            g = self.create_graph()
            progress_bar.update(3, text="Processing communities for the graph")
            self.describe_communities(g)
            print("Graph done - ✅\n")
            print("Starting to create the local and global databases")
            progress_bar.update(4, text="Embeddings of the graph elements")
            self.save_vb_local()
            self.save_vb_global()
            print("Indexation done - ✅\n")
            progress_bar.clear()
        else:
            print("Extraction already done - ✅\n")
