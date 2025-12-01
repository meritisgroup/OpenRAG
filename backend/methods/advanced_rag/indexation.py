from ...database.database_class import DataBase
from ...utils.agent import Agent
from ...utils.splitter import get_splitter
from ...database.data_extraction import DocumentText
from ...database.rag_classes import Document
from .Processor_chunks import Processor_chunks
from ...utils.progress import ProgressBar
from ..naive_rag.indexation import indexation
from ...utils.threading_utils import get_executor_threads
from tqdm.auto import tqdm
import numpy as np
import os
from pathlib import Path
import concurrent.futures



def process_single_doc(
    data_manager,
    processor_chunks,
    model: str,
    path_doc: str,
    doc_index: int,
    config_server: dict,
    agent,
    splitter, 
    chunk_size: int,
    chunk_overlap: bool,
    reset_preprocess: bool,
    use_batch: bool,
) -> dict:
    
    doc_indexation_tokens = 0
    doc = DocumentText(path=path_doc,
                       doc_index=doc_index,
                       config_server=config_server,
                       agent=agent,
                       splitter=splitter,
                       reset_preprocess=reset_preprocess)

    doc_chunks = doc.chunks(chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap)
    name_docs = [str(Path(path_doc).name) for i in range(len(doc_chunks))]
    path_docs = [str(Path(path_doc).parent) for i in range(len(doc_chunks))]
    data = processor_chunks.process_chunk(chunks=doc_chunks,
                                          doc_content=doc.content,
                                          model=model)
    doc_chunks = data["chunks"]

    doc_indexation_tokens += np.sum(data["nb_output_tokens"])
    doc_indexation_tokens += np.sum(indexation(data_manager=data_manager,
                                               doc_chunks=doc_chunks,
                                               path_docs=path_docs))

    return {
        "name": str(Path(path_doc).name),
        "path": str(path_doc),
        "embedding_tokens": int(doc_indexation_tokens),
        "parent_path": str(Path(path_doc).parent),
    }


class AdvancedIndexation:
    def __init__(
        self,
        data_manager,
        agent: Agent,
        embedding_model: str,
        llm_model: str,
        data_preprocessing: str,
        language: str = "EN",
        type_text_splitter: str = "TextSplitter",
        type_processor_chunks: list[str] = [],
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
        self.llm_model = llm_model
        self.language = language

        self.splitter = get_splitter(
            type_text_splitter=type_text_splitter,
            data_preprocessing = data_preprocessing,
            agent=self.agent,
            embedding_model=self.embedding_model,
        )
        self.processor_chunks = Processor_chunks(agent=self.agent,
                                                 type_processor_chunks=type_processor_chunks,
                                                 language=self.language)


    def run_pipeline(
        self,
        config_server,
        chunk_size: int = 1024,
        chunk_overlap: bool = True,
        batch: bool = True,
        reset_preprocess = False,
        max_workers: int = 10
    ) -> None:
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
                                self.processor_chunks,
                                self.llm_model,
                                path_doc,
                                i,
                                config_server,
                                self.agent,
                                self.splitter,
                                chunk_size,
                                chunk_overlap,
                                reset_preprocess,
                                batch,
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
                                                path=result["parent_path"])
        progress_bar.clear()
