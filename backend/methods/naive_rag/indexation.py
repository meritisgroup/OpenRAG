from ...utils.splitter import get_splitter
from ...database.data_extraction import DocumentText
from ...database.rag_classes import Document
from tqdm.auto import tqdm
from ...utils.progress import ProgressBar
import os
import numpy as np
import concurrent.futures
from pathlib import Path
import time
from ...database.rag_classes import Chunk
from ...utils.threading_utils import get_executor_threads



from tqdm import tqdm
import os
import json

def indexation(data_manager, doc_chunks, path_docs):
        """
        Adds a batch of chunks from doc_chunks to the indexation vectorbase
        Args:
            doc_chunks (list[str]) : Chunks to be indexed
            name_docs (list[str]) : Name of docs each chunk is from

        Returns
            None
        """

        tokens = 0
        taille_batch = 500
        range_chunks = range(0, len(doc_chunks), taille_batch)
        for i in range_chunks:
            tokens += np.sum(
                data_manager.add_str_batch_elements(
                    chunks=doc_chunks[i : i + taille_batch],
                    path_docs=path_docs[i : i + taille_batch],
                    display_message=False,
                )
            )
        return tokens


LIST_CHUNKS=[]

def process_single_doc(data_manager,
                       path_doc: str,
                       doc_index: int,
                       config_server: dict,
                       agent,
                       splitter, 
                       chunk_size: int,
                       chunk_overlap: bool,
                       reset_preprocess: bool) -> dict:
    
    doc = DocumentText(path=path_doc,
                       doc_index=doc_index,
                       config_server=config_server,
                       agent=agent,
                       splitter=splitter,
                       reset_preprocess=reset_preprocess)
    
    doc_chunks = doc.chunks(chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap)
    
    LIST_CHUNKS.extend(doc_chunks)
    path_docs = [str(Path(path_doc).parent)] * len(doc_chunks)

    doc_indexation_tokens = 0
    doc_indexation_tokens = indexation(data_manager=data_manager,
                                       doc_chunks=doc_chunks,
                                       path_docs=path_docs)
    
                                       
    return {
        "name": str(Path(path_doc).name),
        "path": str(path_doc),
        "embedding_tokens": int(doc_indexation_tokens),
        "parent_path": str(Path(path_doc).parent),
    }


class NaiveRagIndexation:
    def __init__(
        self,
        data_manager,
        agent,
        embedding_model,
        data_preprocessing: str,
        type_text_splitter="TextSplitter",
    ) -> None:
        """
        Args:
            data_path (str) : path of the folder containing texts you want to have a RAG on
            db (DataBase) : database from ContextualRetrievalRagAgent
            vb (VectorBase) : vectorbase from ContextualRetrievalRagAgent
            model (str) :  Model used to generate chunk context
            language (str) : language the prompts will be written in ("FR" and "EN" available)
            params_host_llm(dict): parameters for Ollama or VLLM, to be set in backend/config_server.json file

        Returns:
            None
        """
        self.data_manager = data_manager
        self.agent = agent
        self.embedding_model = embedding_model

        if type(self.embedding_model) == list:
            embedding_model = self.embedding_model[0]
        else:
            embedding_model = self.embedding_model
            
        self.splitter = get_splitter(type_text_splitter=type_text_splitter,
                                     data_preprocessing=data_preprocessing,
                                     agent=self.agent,
                                     embedding_model=embedding_model)


    def run_pipeline(self,
                     config_server,
                     reset_preprocess: bool = False,
                     chunk_size: int = 1024,
                     chunk_overlap: bool = True,
                     max_workers: int = 1) -> None:
        """
        Split texts from self.data_path, embed them and save them in a vector base.

        Args:
            chunk_size (int): Size of chunks for text splitting.
            chunk_overlap (bool): True if you want the end of the chunk n-1 be the beginning of the chunk n.

        Returns:
            None
        """

        docs_already_processed = [
            res[0] for res in self.data_manager.query(Document.path)
        ]

        to_process_norm = [
            Path(p).resolve().as_posix()
            for p in self.data_manager.get_list_path_documents()
        ]
        docs_already_norm = [
            Path(p).resolve().as_posix() for p in docs_already_processed
        ]
        docs_to_process = [
            doc for doc in to_process_norm if doc not in docs_already_norm
        ]
        if max_workers<=get_executor_threads():
            max_workers = 1

        self.data_manager.create_collection()
        progress_bar = ProgressBar(total=len(docs_to_process))
        index = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_doc,
                                self.data_manager,
                                path_doc,
                                i,
                                config_server,
                                self.agent,
                                self.splitter,
                                chunk_size,
                                chunk_overlap,
                                reset_preprocess
                            ): path_doc for i, path_doc in enumerate(docs_to_process)
                        }

            for future in concurrent.futures.as_completed(futures):
                    path_doc = futures[future]

                    result = future.result()

                    new_doc = Document(name=result["name"],
                                       path=result["path"],
                                       embedding_tokens=result["embedding_tokens"],
                                       input_tokens=0,
                                       output_tokens=0)

                    progress_bar.update(index)
                    index+=1
                    self.data_manager.add_instance(instance=new_doc,
                                           path=result["parent_path"]
                    )
        progress_bar.clear()
        


        
        jsonl_path: str = "chunks_output.jsonl"
        doc_ids = {}
        num_doc = 0

        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for c in tqdm(LIST_CHUNKS, desc="Écriture des chunks", unit="chunk"):
                record = {col.name: getattr(c, col.name) for col in c.__table__.columns}
                if c.document not in doc_ids:
                    num_doc += 1
                    doc_ids[c.document] = num_doc

                record["doc_id"] = doc_ids[c.document]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return None


def contexts_to_prompts(contexts, docs_name):
    context = ""
    docs_context = []
    for i in range(len(contexts)):
        if contexts[i] not in context:
            context += contexts[i] + "\n[...]\n"

            if len(docs_name) > 0:
                docs_context.append(docs_name[i])
    if len(docs_name) > 0:
        return context[:-7], docs_context
    else:
        return context[:-7]


def concat_chunks(chunk_list: list[Chunk]) -> str:
    context = ""
    for chunk in chunk_list:
        if chunk.text not in context:
            context += chunk.text + "\n[...]\n"
    return context
