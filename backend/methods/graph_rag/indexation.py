from ...database.rag_classes import (
    Relation,
    Community,
    MergeEntityOverall,
)
from ...database.database_class import DataBase
from .extract_entities import extract_entities_relations
from ...database.data_extraction import DocumentText
from .graph_creation import Graph
from .community_description import CommunityDescription
from ...utils.splitter import get_splitter
from ...utils.agent import Agent
from ...database.rag_classes import Document, Tokens, Chunk

from pathlib import Path
from tqdm import tqdm
import numpy as np
from sqlalchemy.orm import Session
import os

from ...utils.factory_name_dataset_vectorbase import get_name
from ...utils.progress import ProgressBar


class GraphRagIndexation:
    def __init__(
        self,
        data_manager,
        storage_path: str,
        agent: Agent,
        type_text_splitter: str,
        data_preprocessing: str,
        embedding_model: str,
        llm_model: str,
        language: str = "EN",
    ):  
        
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.data_manager = data_manager
        self.agent = agent
        self.language = language

        self.splitter = get_splitter(
            type_text_splitter=type_text_splitter,
            data_preprocessing=data_preprocessing,
            embedding_model=embedding_model,
            agent=self.agent,
        )
        self.storage_path = storage_path


    def check_extraction_necessity(self, db_name: str = None):
        need_extraction = False

        docs_already_processed = [res[0] for res in self.data_manager.query(Document.path)]
        to_process_norm = [Path(p).resolve().as_posix() for p in self.data_manager.get_list_path_documents(db_name=db_name)]
        docs_already_norm = [Path(p).resolve().as_posix() for p in docs_already_processed]

        docs_to_process = [
            doc
            for doc in to_process_norm
            if doc not in docs_already_norm
        ]
        if docs_to_process != []:
            need_extraction = True
        return need_extraction, docs_to_process

    def extract_entities(self, 
                         db_name: str, 
                         to_process_norm, 
                         config_server,
                         reset_preprocess,
                         chunk_size: int = 1024,
                         overlap: bool = True):
        """
        Extract entities and relations from texts located in the folder self.data_path and save all the results in the data base self.db
        """
        documents_nb_input_tokens = 0
        documents_nb_output_tokens = 0

        progress_bar = ProgressBar(total=len(to_process_norm))
        for k, path_doc in enumerate(to_process_norm):
            progress_bar.update(k-1,
                                text="Entities extraction - Processing file {}".format(path_doc))

            document = DocumentText(path=path_doc,
                                    doc_index=k,
                                    agent=self.agent,
                                    config_server=config_server,
                                    splitter=self.splitter,
                                    reset_preprocess=reset_preprocess)
            chunks = document.chunks(chunk_size=chunk_size,
                                     chunk_overlap=overlap)
            name_docs = [str(Path(path_doc).name) for i in range(len(chunks))]
            path_docs = [str(Path(path_doc).parent) for i in range(len(chunks))]

            entities, relations, input_tokens, output_tokens = (
                    extract_entities_relations(agent=self.agent,
                                               model = self.llm_model,
                                               chunks=chunks,
                                               doc_name=name_docs[0],
                                               language=self.language)
                )
            documents_nb_input_tokens += np.sum(input_tokens)
            documents_nb_output_tokens += np.sum(output_tokens)

            document.input_tokens = np.sum(input_tokens)
            document.output_tokens = np.sum(output_tokens)

            for instance in entities:
                self.data_manager.add_instance(instance,
                                               db_name=db_name)
            for instance in relations:
                self.data_manager.add_instance(instance,
                                               db_name=db_name)

            document_base = document.convert_in_base()
            self.data_manager.add_instance(document_base,
                                           db_name=db_name)

        progress_bar.update(k, 
                            text="Entities extraction - Processing file {}".format(path_doc))
        
        documents_tokens = Tokens(
            title="documents",
            embedding_tokens=0,
            input_tokens=int(documents_nb_input_tokens),
            output_tokens=int(documents_nb_output_tokens),
        )
        self.data_manager.add_instance(documents_tokens,
                                       db_name=db_name)

    def clean_entities(self, db_name: str):
        """
        If an entity has been retrieved multiple times, it merges all the description in one.
        """
        self.data_manager.create_merged_entities_table(db_name=db_name)

    def create_graph(self, db_name :str):
        """
        Create and clusterize the graph of merged entities thanks to relations found in extract_entities process
        """
        entities = self.data_manager.query(MergeEntityOverall,
                                           db_name=db_name)
        relations = self.data_manager.query(Relation,
                                            db_name=db_name)

        g = Graph(entities, relations)

        g.create_communities()
        g.plot_graph(storage_path=self.storage_path, graph_name="graph_rag")

        return g

    def describe_communities(self, graph: Graph, db_name: str):
        """
        Generate a title and a description for each cluster of the provided graph and save it into the data base
        """
        self.data_manager.add_table(Community,
                                    db_name=db_name)
        db = self.data_manager.get_database(db_name=db_name)
        cd = CommunityDescription(agent=self.agent,
                                  model=self.llm_model,
                                  graph=graph,
                                  db=db)

        cd.process_communities(language=self.language)
        db = cd.get_database()
        self.data_manager.set_database(db_name=db_name,
                                       db=db)

    def save_vb_local(self, db_name: str):
        """
        Save entities' names in a vector base
        """
        entities_nb_tokens = 0
    
        self.data_manager.create_collection(name="local_search",
                                            vb_name=db_name)
        entities = [res[0] for res in self.data_manager.query(MergeEntityOverall.name,
                                                              db_name=db_name)]

        with tqdm(
            range(1 + len(entities) // 10), desc="Embedding for local search"
        ) as progress_bar:

            for k in progress_bar:
                truncated_entities = entities[10 * k : min(10 * (k + 1), len(entities))]

                if truncated_entities != []:
                    nb_tokens = 0
                    taille_batch = 100
                    for i in range(len(truncated_entities)):
                        truncated_entities[i] = Chunk(text=truncated_entities[i], 
                                                      document="",
                                                       id=i + 1)

                    for j in range(0, len(truncated_entities), taille_batch):
                        nb_tokens = np.sum(self.data_manager.add_str_batch_elements(collection_name="local_search",
                                                                                    chunks=truncated_entities[j:j + taille_batch],
                                                                                    display_message=False,
                                                                                    vb_name=db_name
                                                                                    ))

                        entities_nb_tokens += np.sum(nb_tokens)
                if k == len(entities) // 10:
                    progress_bar.set_description("Embedding for local search - ✅")

        entities_tokens = Tokens(title="entities",
                                 embedding_tokens=int(entities_nb_tokens),
                                 input_tokens=0,
                                 output_tokens=0)
        self.data_manager.add_instance(entities_tokens,
                                       db_name=db_name)


    def save_vb_global(self, db_name: str):
        """
        Save ecollection titles in a vector base
        """
        db = self.data_manager.get_database(db_name=db_name)
        communities_embedding_tokens = 0
        community_instance: Tokens = db.get_instance_by_title(Tokens, "communities")

        self.data_manager.create_collection(name="global_search",
                                            vb_name=db_name)
        communities = [res.title for res in db.query(Community)]

        with tqdm(range(1 + len(communities) // 10), desc="Embedding for global search") as progress_bar:
            for k in progress_bar:
                truncated_communities = communities[
                    10 * k : min(10 * (k + 1), len(communities))
                ]

                if truncated_communities != []:
                    nb_tokens = 0
                    taille_batch = 100
                    for i in range(len(truncated_communities)):
                        truncated_communities[i] = Chunk(text=truncated_communities[i], 
                                                             document="",
                                                                id=i + 1)
                    for j in range(0, len(truncated_communities), taille_batch):
                        nb_tokens = np.sum(self.data_manager.add_str_batch_elements(
                                           collection_name="global_search",
                                           chunks=truncated_communities[j:j + taille_batch],
                                           display_message=False,
                                           vb_name=db_name))

                        communities_embedding_tokens += np.sum(nb_tokens)

                if k == len(communities) // 10:
                    progress_bar.set_description("Embedding for global search - ✅")

        community_instance.embedding_tokens = communities_embedding_tokens
        self.data_manager.update_instance(db_name=db_name)
        self.data_manager.set_database(db_name=db_name,
                                       db=db)


    def run_pipeline(self, config_server, chunk_size: int = 1024,
                     overlap: bool = True, reset_preprocess: bool = False):
        """
        Indexation phase for graph rag, from entities extraction to community description.
        """        
        dbs_name = self.data_manager.get_dbs_name()
        for db_name in dbs_name:
            need_extraction, docs_to_process = self.check_extraction_necessity(db_name=db_name)

            if need_extraction:
                progress_bar = ProgressBar(total=5)
                progress_bar.update(0, text="Extraction of entities and relations\n")
                self.extract_entities(chunk_size=chunk_size,
                                      to_process_norm=docs_to_process, 
                                        overlap=overlap,
                                        db_name=db_name,
                                        config_server=config_server,
                                        reset_preprocess=reset_preprocess)
                progress_bar.update(1, text="Merging muliples entities of entities")
                self.clean_entities(db_name=db_name)
                progress_bar.update(2, text="Creation of the graph")
                g = self.create_graph(db_name=db_name)
                progress_bar.update(3, text="Processing communities for the graph")
                self.describe_communities(g, db_name=db_name)
                progress_bar.update(4, text="Embeddings of the graph elements")
                self.save_vb_local(db_name=db_name)
                self.save_vb_global(db_name=db_name)
                progress_bar.clear()
