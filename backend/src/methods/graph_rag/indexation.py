from database.rag_classes import Relation, Community, MergeEntityOverall
from database.database_class import DataBase
from .extract_entities import extract_entities_relations
from database.data_extraction import DocumentText
from .graph_creation import Graph
from .community_description import CommunityDescription
from utils.splitter import get_splitter
from utils.agent import Agent
from database.rag_classes import Document, Tokens, Chunk
from pathlib import Path
from utils.progress import ProgressBar, tqdm, TwoLevelProgressTracker
import numpy as np
from sqlalchemy.orm import Session
import os

class GraphRagIndexation:

    def __init__(self, data_manager, storage_path: str, agent: Agent, type_text_splitter: str, data_preprocessing: str, embedding_model: str, llm_model: str, language: str='EN'):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.data_manager = data_manager
        self.agent = agent
        self.language = language
        self.splitter = get_splitter(type_text_splitter=type_text_splitter, data_preprocessing=data_preprocessing, embedding_model=embedding_model, agent=self.agent)
        self.storage_path = storage_path

    def check_extraction_necessity(self, db_name: str=None):
        need_extraction = False
        docs_already_processed = [res[0] for res in self.data_manager.query(Document.path)]
        to_process_norm = [Path(p).resolve().as_posix() for p in self.data_manager.get_list_path_documents(db_name=db_name)]
        docs_already_norm = [Path(p).resolve().as_posix() for p in docs_already_processed]
        docs_to_process = [doc for doc in to_process_norm if doc not in docs_already_norm]
        if docs_to_process != []:
            need_extraction = True
        return (need_extraction, docs_to_process)

    def extract_entities(self, db_name: str, to_process_norm, config_server, reset_preprocess, chunk_size: int=1024, overlap: bool=True):
        documents_nb_input_tokens = 0
        documents_nb_output_tokens = 0
        progress_bar = ProgressBar(total=len(to_process_norm))
        for (k, path_doc) in enumerate(to_process_norm):
            progress_bar.update(k - 1, text='Entities extraction - Processing file {}'.format(path_doc))
            document = DocumentText(path=path_doc, doc_index=k, agent=self.agent, config_server=config_server, splitter=self.splitter, reset_preprocess=reset_preprocess)
            chunks = document.chunks(chunk_size=chunk_size, chunk_overlap=overlap)
            name_docs = [str(Path(path_doc).name) for i in range(len(chunks))]
            path_docs = [str(Path(path_doc).parent) for i in range(len(chunks))]
            (entities, relations, input_tokens, output_tokens) = extract_entities_relations(agent=self.agent, model=self.llm_model, chunks=chunks, doc_name=name_docs[0], language=self.language)
            documents_nb_input_tokens += np.sum(input_tokens)
            documents_nb_output_tokens += np.sum(output_tokens)
            document.input_tokens = np.sum(input_tokens)
            document.output_tokens = np.sum(output_tokens)
            for instance in entities:
                self.data_manager.add_instance(instance, db_name=db_name)
            for instance in relations:
                self.data_manager.add_instance(instance, db_name=db_name)
            document_base = document.convert_in_base()
            self.data_manager.add_instance(document_base, db_name=db_name)
        progress_bar.update(k, text='Entities extraction - Processing file {}'.format(path_doc))
        documents_tokens = Tokens(title='documents', embedding_tokens=0, input_tokens=int(documents_nb_input_tokens), output_tokens=int(documents_nb_output_tokens))
        self.data_manager.add_instance(documents_tokens, db_name=db_name)

    def clean_entities(self, db_name: str):
        self.data_manager.create_merged_entities_table(db_name=db_name)

    def extract_entities_with_progress(self, db_name: str, to_process_norm, config_server, reset_preprocess, chunk_size: int=1024, overlap: bool=True, tracker=None):
        documents_nb_input_tokens = 0
        documents_nb_output_tokens = 0
        
        for (k, path_doc) in enumerate(to_process_norm):
            if tracker:
                doc_name = Path(path_doc).name
                tracker.update_sub(k + 1, f"Extracting entities ({k+1}/{len(to_process_norm)}) - {doc_name}")
            
            document = DocumentText(path=path_doc, doc_index=k, agent=self.agent, config_server=config_server, splitter=self.splitter, reset_preprocess=reset_preprocess)
            chunks = document.chunks(chunk_size=chunk_size, chunk_overlap=overlap)
            name_docs = [str(Path(path_doc).name) for i in range(len(chunks))]
            path_docs = [str(Path(path_doc).parent) for i in range(len(chunks))]
            (entities, relations, input_tokens, output_tokens) = extract_entities_relations(agent=self.agent, model=self.llm_model, chunks=chunks, doc_name=name_docs[0], language=self.language)
            documents_nb_input_tokens += np.sum(input_tokens)
            documents_nb_output_tokens += np.sum(output_tokens)
            for instance in entities:
                self.data_manager.add_instance(instance, db_name=db_name)
            for instance in relations:
                self.data_manager.add_instance(instance, db_name=db_name)
            document_base = document.convert_in_base()
            self.data_manager.add_instance(document_base, db_name=db_name)
        
        documents_tokens = Tokens(title='documents', embedding_tokens=0, input_tokens=int(documents_nb_input_tokens), output_tokens=int(documents_nb_output_tokens))
        self.data_manager.add_instance(documents_tokens, db_name=db_name)

    def create_graph(self, db_name: str):
        entities = list(self.data_manager.query(MergeEntityOverall, db_name=db_name))
        relations = list(self.data_manager.query(Relation, db_name=db_name))
        g = Graph(entities, relations)
        g.create_communities()
        g.plot_graph(storage_path=self.storage_path, graph_name='graph_rag')
        return g

    def describe_communities(self, graph: Graph, db_name: str):
        self.data_manager.add_table(Community, db_name=db_name)
        db = self.data_manager.get_database(db_name=db_name)
        cd = CommunityDescription(agent=self.agent, model=self.llm_model, graph=graph, db=db)
        cd.process_communities(language=self.language)
        db = cd.get_database()
        self.data_manager.set_database(db_name=db_name, db=db)

    def save_vb_local(self, db_name: str, tracker=None):
        entities_nb_tokens = 0
        self.data_manager.create_collection(name='graph_rag_local', vb_name=db_name)
        entities = [res[0] for res in self.data_manager.query(MergeEntityOverall.name, db_name=db_name)]
        if tracker:
            tracker.set_sub_total(len(entities) // 10 + 1)
        with tqdm(range(1 + len(entities) // 10), desc='Embedding for local search') as progress_bar:
            for k in progress_bar:
                if tracker:
                    tracker.update_sub(k + 1, f"Embedding entities for local search ({k+1}/{len(entities)//10 + 1})")
                truncated_entities = entities[10 * k:min(10 * (k + 1), len(entities))]
                if truncated_entities != []:
                    nb_tokens = 0
                    taille_batch = 100
                    for i in range(len(truncated_entities)):
                        truncated_entities[i] = Chunk(text=truncated_entities[i], document='', id=i + 1)
                    for j in range(0, len(truncated_entities), taille_batch):
                        nb_tokens = np.sum(self.data_manager.add_str_batch_elements(collection_name='graph_rag_local', chunks=truncated_entities[j:j + taille_batch], display_message=False, vb_name=db_name))
                        entities_nb_tokens += np.sum(nb_tokens)
                if k == len(entities) // 10:
                    progress_bar.set_description('Embedding for local search - ✅')
        entities_tokens = Tokens(title='entities', embedding_tokens=int(entities_nb_tokens), input_tokens=0, output_tokens=0)
        self.data_manager.add_instance(entities_tokens, db_name=db_name)

    def save_vb_global(self, db_name: str, tracker=None):
        db = self.data_manager.get_database(db_name=db_name)
        communities_embedding_tokens = 0
        community_instance: Tokens = db.get_instance_by_title(Tokens, 'communities')
        is_new_instance = False
        if community_instance is None:
            community_instance = Tokens(title='communities', embedding_tokens=0, input_tokens=0, output_tokens=0)
            is_new_instance = True
        self.data_manager.create_collection(name='graph_rag_global', vb_name=db_name)
        communities = [res.title for res in list(db.query(Community))]
        if tracker:
            tracker.set_sub_total(len(communities) // 10 + 1)
        print(f'[DEBUG] save_vb_global: Found {len(communities)} communities to embed for db_name={db_name}')
        if communities:
            print(f'[DEBUG] save_vb_global: First few communities: {communities[:3]}')
        with tqdm(range(1 + len(communities) // 10), desc='Embedding for global search') as progress_bar:
            for k in progress_bar:
                if tracker:
                    tracker.update_sub(k + 1, f"Embedding communities for global search ({k+1}/{len(communities)//10 + 1})")
                truncated_communities = communities[10 * k:min(10 * (k + 1), len(communities))]
                if truncated_communities != []:
                    nb_tokens = 0
                    taille_batch = 100
                    for i in range(len(truncated_communities)):
                        truncated_communities[i] = Chunk(text=truncated_communities[i], document='', id=i + 1)
                    for j in range(0, len(truncated_communities), taille_batch):
                        print(f'[DEBUG] save_vb_global: Adding batch {j}-{j+taille_batch} ({len(truncated_communities[j:j + taille_batch])} chunks) to collection graph_rag_global')
                        nb_tokens = np.sum(self.data_manager.add_str_batch_elements(collection_name='graph_rag_global', chunks=truncated_communities[j:j + taille_batch], display_message=False, vb_name=db_name))
                        communities_embedding_tokens += np.sum(nb_tokens)
                        print(f'[DEBUG] save_vb_global: Batch added, tokens={nb_tokens}')
                if k == len(communities) // 10:
                    progress_bar.set_description('Embedding for global search - ✅')
        community_instance.embedding_tokens = communities_embedding_tokens
        if is_new_instance:
            self.data_manager.add_instance(community_instance, db_name=db_name)
        else:
            self.data_manager.update_instance(db_name=db_name)
        self.data_manager.set_database(db_name=db_name, db=db)

    def run_pipeline(self, config_server, chunk_size: int=1024, overlap: bool=True, reset_preprocess: bool=False, progress_callback=None):
        dbs_name = self.data_manager.get_dbs_name()
        
        # Calculer total des étapes (5 par DB)
        total_steps = len(dbs_name) * 5
        
        # Créer tracker si callback fourni
        tracker = None
        if progress_callback:
            tracker = TwoLevelProgressTracker(total_steps, progress_callback)
            tracker.update_global(f"Starting GraphRAG indexation ({len(dbs_name)} database(s))...")
        
        for db_name in dbs_name:
            (need_extraction, docs_to_process) = self.check_extraction_necessity(db_name=db_name)
            
            if need_extraction:
                # Étape 1: Extraction (avec sous-progression par document)
                if tracker:
                    tracker.set_sub_total(len(docs_to_process))
                
                self.extract_entities_with_progress(
                    chunk_size=chunk_size,
                    to_process_norm=docs_to_process,
                    overlap=overlap,
                    db_name=db_name,
                    config_server=config_server,
                    reset_preprocess=reset_preprocess,
                    tracker=tracker
                )
                
                if tracker:
                    tracker.complete_step(f"Extraction completed ({db_name})")
                
                # Étape 2: Clean entities
                if tracker:
                    tracker.update_global(f"Merging entities ({db_name})...")
                self.clean_entities(db_name=db_name)
                if tracker:
                    tracker.complete_step(f"Merging completed ({db_name})")
                
                # Étape 3: Create graph
                if tracker:
                    tracker.update_global(f"Creating graph ({db_name})...")
                g = self.create_graph(db_name=db_name)
                if tracker:
                    tracker.complete_step(f"Graph created ({db_name})")
                
                # Étape 4: Describe communities
                if tracker:
                    tracker.update_global(f"Processing communities ({db_name})...")
                self.describe_communities(g, db_name=db_name)
                if tracker:
                    tracker.complete_step(f"Communities processed ({db_name})")
                
                # Étape 5: Embeddings (local + global avec sous-progression)
                if tracker:
                    tracker.update_global(f"Creating embeddings ({db_name})...")
                
                self.save_vb_local(db_name=db_name, tracker=tracker)
                self.save_vb_global(db_name=db_name, tracker=tracker)
                
                if tracker:
                    tracker.complete_step(f"Embeddings completed ({db_name})")
            else:
                # Skip: déjà indexé - avancer toutes les étapes
                if tracker:
                    for _ in range(5):
                        tracker.complete_step(f"Already indexed ({db_name}), skipping...")
        
        if tracker:
            tracker.complete_all()