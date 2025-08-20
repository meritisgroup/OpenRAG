from ...database.database_class import DataBase
from ...utils.agent import Agent
from ...utils.splitter import get_splitter
from ..graph_rag.extract_entities import DocumentText
from ...database.rag_classes import Document
from .Processor_chunks import Processor_chunks
from tqdm.auto import tqdm
import numpy as np
import os
from pathlib import Path


class NaiveRagIndexation:
    def __init__(
        self,
        data_manager,
        agent: Agent,
        embedding_model,
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
        self.language = language

        self.splitter = get_splitter(
            type_text_splitter=type_text_splitter,
            agent=self.agent,
            embedding_model=self.embedding_model,
        )
        self.processor_chunks = Processor_chunks(
            agent=self.agent,
            type_processor_chunks=type_processor_chunks,
            language=self.language,
        )

    def __batch_indexation__(self, doc_chunks, name_docs, path_docs):
        """
        Adds a batch of chunks from doc_chunks to the indexation verctorbase
        Args:
            doc_chunks (list[str]) : Chunks to be indexed
            name_docs (list[str]) : Name of docs each chunk is from

        Returns
            None
        """
        elements = []
        for k, chunk in enumerate(doc_chunks):
            elements.append(chunk.text.replace("\n", "").replace("'", ""))

        tokens = 0
        taille_batch = 500
        for i in range(0, len(elements), taille_batch):
            tokens += np.sum(self.data_manager.add_str_batch_elements(
                    elements=elements[i:i + taille_batch],
                    docs_name=name_docs[i : i + taille_batch],
                    path_docs=path_docs[i : i + taille_batch],
                    display_message=False
            ))
        return tokens

    def __serial_indexation__(self, doc_chunks, name_docs, path_docs) -> int:
        """
        Adds a batch of chunks from doc_chunks to the indexation verctorbase
        Args:
            doc_chunks (list[str]) : Chunks to be indexed
            name_docs (list[str]) : Name of docs each chunk is from

        Returns
            indexation_tokens :
        """
        indexation_tokens = 0
        for k, chunk in enumerate(doc_chunks):
            try:
                indexation_tokens += self.data_manager.add_str_elements(
                    elements=[chunk.text.replace("\n", "").replace("'", "")],
                    docs_name=[name_docs[k]],
                    path_docs=[path_docs[k]],
                    display_message=False,
                )
            except:
                None
        return indexation_tokens

    def run_pipeline(
        self, chunk_size: int = 500, chunk_overlap: bool = True, batch: bool = True
    ) -> None:
        """
        Split texts from self.data_path, embed them and save them in a vector base.

        Args:
            chunk_size (int): Size of chunks for text splitting.
            chunk_overlap (bool): True if you want the end of the chunk n-1 be the beginning of the chunk n.

        Returns:
            None
        """
        docs_already_processed = [res[0] for res in self.data_manager.query(Document.path)]
        to_process_norm = [Path(p).resolve().as_posix() for p in self.data_manager.get_list_path_documents()]
        docs_already_norm = [Path(p).resolve().as_posix() for p in docs_already_processed]

        docs_to_process = [
            doc
            for doc in to_process_norm
            if doc not in docs_already_norm
        ]

        self.data_manager.create_collection()
        with tqdm(docs_to_process) as progress_bar:
            for i, path_doc in enumerate(progress_bar):
                doc_tokens = 0
                progress_bar.set_description(f"Embbeding chunks - {path_doc}")
                doc = DocumentText(
                    path=path_doc, splitter=self.splitter
                )
                doc_chunks = doc.chunks(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                name_docs = [str(Path(path_doc).name) for i in range(len(doc_chunks))]
                path_docs = [str(Path(path_doc).parent) for i in range(len(doc_chunks))]
                try:
                    data = self.processor_chunks.process_chunk(
                        chunks=doc_chunks, doc_content=doc.content, batch=batch
                    )
                    doc_chunks = data["chunks"]
                    doc_tokens+=np.sum(data["nb_output_tokens"])
                    if batch:
                        doc_tokens += self.__batch_indexation__(
                            doc_chunks=doc_chunks, name_docs=name_docs, path_docs=path_docs
                        )

                    else:
                        doc_tokens += self.__serial_indexation__(
                            doc_chunks=doc_chunks, name_docs=name_docs, path_docs=path_docs
                        )
                except Exception:
                    print("Failed indexing: {}".format(path_doc))

                if i == len(progress_bar) - 1:
                    progress_bar.set_description(f"Embbeding chunks - ✅")

                new_doc = Document(
                    name=str(Path(path_doc).name),
                                path=str(Path(path_doc)),
                    embedding_tokens=doc_tokens,
                    input_tokens=0,
                    output_tokens=0,
                )
                self.data_manager.add_instance(new_doc,
                                               path=str(Path(path_doc).parent))
