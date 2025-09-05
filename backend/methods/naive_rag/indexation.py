from ...utils.splitter import get_splitter
from ...database.data_extraction import DocumentText
from ...database.rag_classes import Document
from tqdm.auto import tqdm
from ...utils.progress import ProgressBar
import os
import numpy as np
from pathlib import Path

class NaiveRagIndexation:
    def __init__(
        self,
        data_manager,
        agent,
        embedding_model,
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

        self.splitter = get_splitter(
            type_text_splitter=type_text_splitter,
            agent=self.agent,
            embedding_model=self.embedding_model,
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
        tokens = 0
        for k, chunk in enumerate(doc_chunks):
            elements.append(chunk.text.replace("\n", "").replace("'", ""))

        tokens = 0
        taille_batch = 500
        range_chunks = range(0, len(elements), taille_batch)
        progress_bar_chunks = ProgressBar(total=len(range_chunks))
        j = 0
        for i in range_chunks:
            tokens += np.sum(
                self.data_manager.add_str_batch_elements(elements=elements[i : i + taille_batch],
                                                         docs_name=name_docs[i : i + taille_batch],
                                                         path_docs=path_docs[i : i + taille_batch],
                                                         display_message=False))
            progress_bar_chunks.update(j)
            j+=1
        progress_bar_chunks.clear()
        return tokens
    

    def __serial_indexation__(self, doc_chunks, name_docs, path_docs):
        """
        Adds a batch of chunks from doc_chunks to the indexation verctorbase
        Args:
            doc_chunks (list[str]) : Chunks to be indexed
            name_docs (list[str]) : Name of docs each chunk is from

        Returns
            None
        """
        tokens = 0
        for k, chunk in enumerate(doc_chunks):
            try:
                tokens += self.data_manager.add_str_elements(elements=[chunk.text.replace("\n", "").replace("'", "")],
                                                             docs_name=[name_docs[k]],
                                                             path_docs=[path_docs[k]],
                                                             display_message=False)
            except Exception:
                None
        return tokens
    

    def run_pipeline(
        self, chunk_size: int = 500, chunk_overlap: bool = True, batch: bool = True, config_server={}
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
        progress_bar = ProgressBar(total=len(docs_to_process))
        for i, path_doc in enumerate(docs_to_process):
            doc_indexation_tokens = 0
            progress_bar.set_description(f"Embbeding chunks - {path_doc}")
            doc = DocumentText(path=path_doc, 
                                doc_index=i,
                                config_server=config_server,
                                splitter=self.splitter)
                
            doc_chunks = doc.chunks(chunk_size=chunk_size,
                                        chunk_overlap=chunk_overlap)
    
            name_docs = [str(Path(path_doc).name) for i in range(len(doc_chunks))]
            path_docs = [str(Path(path_doc).parent) for i in range(len(doc_chunks))]

            if batch:
                doc_indexation_tokens += self.__batch_indexation__(doc_chunks=doc_chunks,
                                                                            name_docs=name_docs,
                                                                            path_docs=path_docs)

            else:
                doc_indexation_tokens += self.__serial_indexation__(doc_chunks=doc_chunks, 
                                                                                name_docs=name_docs,
                                                                                path_docs=path_docs)

            new_doc = Document(
                                    name=str(Path(path_doc).name),
                                    path=str(Path(path_doc)),
                                    embedding_tokens=int(doc_indexation_tokens),
                                    input_tokens=0,
                                    output_tokens=0,
                                )
            self.data_manager.add_instance(instance=new_doc,
                                                path=str(Path(path_doc).parent))
            progress_bar.update(i)
        progress_bar.clear()



def contexts_to_prompts(contexts, docs_name):
    context = ""
    docs_context = []
    for i in range(len(contexts)):
        if contexts[i] not in context:
            context += contexts[i] + "\n[...]\n"

            if len(docs_name)>0:
                docs_context.append(docs_name[i])
    if len(docs_name)>0:
        return context[:-7], docs_context
    else:
        return context[:-7]