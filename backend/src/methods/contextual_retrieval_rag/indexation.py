from database.data_extraction import DocumentText
from database.rag_classes import Document, Chunk
from utils.splitter import get_splitter
from .contextual import run_contextual
from utils.progress import ProgressBar, tqdm
import os
import numpy as np
from .prompts import prompts
from methods.naive_rag.indexation import indexation
from utils.threading_utils import get_executor_threads
from pathlib import Path
import concurrent.futures

def process_single_doc(data_manager, agent, model: str, path_doc: str, doc_index: int, config_server: dict, splitter, chunk_size: int, chunk_overlap: bool, reset_preprocess: bool, size_limit: int, language: str) -> dict:
    doc_indexation_tokens = 0
    input_tokens = 0
    output_tokens = 0
    doc = DocumentText(path=path_doc, doc_index=doc_index, config_server=config_server, agent=agent, splitter=splitter, reset_preprocess=reset_preprocess)
    doc_content = doc.content
    size_limit_doc = []
    left = 0
    while left + size_limit < len(doc_content):
        size_limit_doc.append(doc_content[left:left + size_limit])
        left += size_limit
    size_limit_doc.append(doc_content[left:])
    splitter = splitter
    progress_bar_doc = ProgressBar(total=len(size_limit_doc))
    for (j, little_doc) in enumerate(size_limit_doc):
        doc_chunks = []
        name_docs = []
        chunked_docs = splitter.split_text(little_doc)
        for k in range(len(chunked_docs)):
            doc_chunks.append(Chunk(text=chunked_docs[k], document=path_doc + f'_{j}', id=k))
            name_docs.append(str(Path(path_doc).name) + f'_{j}')
        name_docs = [str(Path(path_doc).name) for i in range(len(doc_chunks))]
        path_docs = [str(Path(path_doc).parent) for i in range(len(doc_chunks))]
        chunk_with_context = run_contextual(agent=agent, doc_chunks=doc_chunks, model=model, doc_content=little_doc, language=language)
        chunk_with_context = chunk_with_context
        input_tokens += chunk_with_context['nb_input_tokens']
        output_tokens += chunk_with_context['nb_output_tokens']
        doc_indexation_tokens += indexation(doc_chunks=chunk_with_context['texts'], path_docs=path_docs)
    return {'name': str(Path(path_doc).name), 'path': str(path_doc), 'embedding_tokens': int(doc_indexation_tokens), 'input_tokens': int(input_tokens), 'output_tokens': int(output_tokens), 'parent_path': str(Path(path_doc).parent)}

class ContextualRetrievalIndexation:

    def __init__(self, data_manager, language: str, agent, embedding_model: str, data_preprocessing: str, type_text_splitter='TextSplitter') -> None:
        self.data_manager = data_manager
        self.language = language
        self.agent = agent
        self.prompts = prompts[language]
        self.splitter = get_splitter(type_text_splitter=type_text_splitter, data_preprocessing=data_preprocessing, agent=self.agent, embedding_model=embedding_model)
        self.input_tokens = 0
        self.output_tokens = 0

    def __indexation__(self, doc_chunks: list[str], path_docs):
        doc_tokens = 0
        elements = []
        for (k, chunk) in enumerate(doc_chunks):
            elements.append(chunk.text.replace('\n', '').replace("'", ''))
        taille_batch = 100
        range_chunks = range(0, len(elements), taille_batch)
        progress_bar_chunks = ProgressBar(total=len(range_chunks))
        j = 0
        for i in range_chunks:
            doc_tokens += np.sum(self.data_manager.add_str_batch_elements(chunks=doc_chunks[i:i + taille_batch], path_docs=path_docs[i:i + taille_batch], display_message=False))
            progress_bar_chunks.update(j)
            j += 1
        progress_bar_chunks.clear()
        return doc_tokens

    def run_pipeline(self, config_server, model: str, chunk_size: int=1024, chunk_overlap: bool=True, reset_preprocess=False, max_workers: int=10, size_limit: int=16000) -> None:
        docs_already_processed = [res[0] for res in self.data_manager.query(Document.path)]
        to_process_norm = [Path(p).resolve().as_posix() for p in self.data_manager.get_list_path_documents()]
        docs_already_norm = [Path(p).resolve().as_posix() for p in docs_already_processed]
        docs_to_process = [doc for doc in to_process_norm if doc not in docs_already_norm]
        if max_workers <= get_executor_threads():
            max_workers = 1
        self.data_manager.create_collection()
        progress_bar = ProgressBar(total=len(docs_to_process))
        index = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_doc, self.data_manager, self.agent, path_doc, model, i, config_server, self.splitter, chunk_size, chunk_overlap, reset_preprocess): path_doc for (i, path_doc) in enumerate(docs_to_process)}
        for future in concurrent.futures.as_completed(futures):
            path_doc = futures[future]
            result = future.result()
            new_doc = Document(name=result['name'], path=result['path'], embedding_tokens=result['embedding_tokens'], input_tokens=result['input_tokens'], output_tokens=result['output_tokens'])
            progress_bar.update(index)
            index += 1
            self.data_manager.add_instance(instance=new_doc, path=result['parent_path'])
        progress_bar.clear()