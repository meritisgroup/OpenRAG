import os
import pickle
import shutil
import numpy as np
import faiss
from typing import Union, Type
from .base_classes import VectorBase
from database.rag_classes import Chunk


class VectorBase_embeddings_faiss(VectorBase):

    def __init__(self, vb_name: str, storage_path: str, agent, embedding_model: str):
        self.vb_name = vb_name
        self.agent = agent
        self.embedding_model = embedding_model
        self.nb_tokens_embeddings = 0
        self.storage_path = storage_path
        self.base_dir = os.path.join(storage_path, 'faiss')
        os.makedirs(self.base_dir, exist_ok=True)
        self.dimension = len(self.agent.embeddings('test', model=embedding_model)['embeddings'][0])
        self._collections = {}
        self._try_load_collection(self.vb_name)

    def _collection_dir(self, name: str) -> str:
        return os.path.join(self.base_dir, name)

    def _try_load_collection(self, name: str) -> bool:
        cdir = self._collection_dir(name)
        index_path = os.path.join(cdir, 'index.faiss')
        meta_path = os.path.join(cdir, 'metadata.pkl')
        if os.path.exists(index_path) and os.path.exists(meta_path):
            index = faiss.read_index(index_path)
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
            next_id = max(metadata.keys()) + 1 if metadata else 0
            self._collections[name] = {
                'index': index,
                'metadata': metadata,
                'next_id': next_id
            }
            return True
        return False

    def _save_collection(self, name: str) -> None:
        cdir = self._collection_dir(name)
        os.makedirs(cdir, exist_ok=True)
        col = self._collections[name]
        faiss.write_index(col['index'], os.path.join(cdir, 'index.faiss'))
        with open(os.path.join(cdir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(col['metadata'], f)

    def build_connection(self):
        pass

    def create_collection(self, name=None, add_fields=[]) -> None:
        if name is None:
            name = self.vb_name
        if name in self._collections:
            print(f'The collection "{name}" already exists')
            return
        if self._try_load_collection(name):
            print(f'The collection "{name}" already exists')
            return
        index = faiss.IndexFlatIP(self.dimension)
        self._collections[name] = {'index': index, 'metadata': {}, 'next_id': 0}

    def check_collection_exist(self, collection_name) -> bool:
        if collection_name in self._collections:
            return True
        return self._try_load_collection(collection_name)

    def check_element_exist(self, element, collection_name=None) -> bool:
        if collection_name is None:
            collection_name = self.vb_name
        if collection_name not in self._collections:
            return False
        for chunk_data in self._collections[collection_name]['metadata'].values():
            if chunk_data.get('text') == element:
                return True
        return False

    def add_str_batch_elements(self, chunks: list = [], display_message: bool = True,
                               collection_name=None) -> int:
        if collection_name is None:
            collection_name = self.vb_name
        if not self.check_collection_exist(collection_name):
            self.create_collection(name=collection_name)
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        result = self.agent.embeddings(texts=texts, model=self.embedding_model)
        embeddings = np.array(result['embeddings'], dtype='float32')
        faiss.normalize_L2(embeddings)

        nb_tokens = result['nb_tokens']
        if isinstance(nb_tokens, list):
            nb_tokens = int(np.sum(nb_tokens))

        col = self._collections[collection_name]
        start_id = col['next_id']
        col['index'].add(embeddings)

        for i, chunk in enumerate(chunks):
            chunk_dict = {}
            for c in chunk.__table__.columns:
                chunk_dict[c.name] = getattr(chunk, c.name)
            col['metadata'][start_id + i] = chunk_dict
        col['next_id'] = start_id + len(chunks)

        self._save_collection(collection_name)

        if display_message:
            print(f'{len(chunks)} elements have been successfully added in the vector base')
        self.nb_tokens_embeddings += nb_tokens
        return nb_tokens

    def add_str_elements(self, chunks: list = [], display_message: bool = True,
                         collection_name=None) -> int:
        return self.add_str_batch_elements(chunks, display_message, collection_name)

    def k_search(self, queries: Union[str, list], k: int, output_fields: list = ['text'],
                 filters: dict = None, collection_name=None, type_output=Chunk):
        if collection_name is None:
            collection_name = self.vb_name
        if isinstance(queries, str):
            queries = [queries]
        if not self.check_collection_exist(collection_name):
            return [[] for _ in queries]

        result = self.agent.embeddings(texts=queries, model=self.embedding_model)
        query_vectors = np.array(result['embeddings'], dtype='float32')
        faiss.normalize_L2(query_vectors)

        col = self._collections[collection_name]
        actual_k = min(k, col['index'].ntotal)
        if actual_k == 0:
            return [[] for _ in queries]

        distances, indices = col['index'].search(query_vectors, actual_k)

        results = []
        valid_columns = {c.name for c in type_output.__table__.columns}
        for i in range(len(queries)):
            row = []
            for j in range(actual_k):
                idx = int(indices[i][j])
                if idx == -1:
                    continue
                chunk_data = col['metadata'].get(idx)
                if chunk_data:
                    filtered = {k_: v for k_, v in chunk_data.items() if k_ in valid_columns}
                    row.append(type_output(**filtered))
            results.append(row)
        return results

    def delete_collection(self, vb_name=None) -> None:
        if vb_name is None:
            vb_name = self.vb_name
        if vb_name in self._collections:
            del self._collections[vb_name]
        cdir = self._collection_dir(vb_name)
        if os.path.exists(cdir):
            shutil.rmtree(cdir)
            print('the collection have been deleted')
        else:
            print('The collection does not exist')

    def add_name_done_doc(self):
        pass

    def get_nb_token_embeddings(self):
        return self.nb_tokens_embeddings
