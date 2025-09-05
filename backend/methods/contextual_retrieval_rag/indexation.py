from ...database.data_extraction import DocumentText
from ...database.rag_classes import Document,Chunk
from ...utils.splitter import get_splitter
from .contextual import run_batch_contextual, run_serial_contextual
from tqdm.auto import tqdm
import os
from ...utils.progress import ProgressBar
import numpy as np
from .prompts import prompts
from pathlib import Path


class ContextualRetrievalIndexation:
    "A class that handles the whole indexation process for the Contextual retrieval rag"

    def __init__(self, 
                 data_manager,
                 language: str,
                 agent,
                 embedding_model :str,
                 type_text_splitter = "TextSplitter") -> None:
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
        self.language = language
        self.agent = agent
        self.prompts = prompts[language]
        self.splitter = get_splitter(type_text_splitter=type_text_splitter,
                                     agent=self.agent, 
                                     embedding_model=embedding_model)
        self.input_tokens = 0
        self.output_tokens = 0

    def __batch_indexation__(self, doc_chunks: list[str], name_docs, path_docs):
        """
        Adds a batch of chunks from doc_chunks to the indexation verctorbase
        Args:
            doc_chunks (list[str]) : Chunks to be indexed
            name_docs (list[str]) : Name of docs each chunk is from

        Returns
            None
        """
        doc_tokens = 0
        elements = []
        for k, chunk in enumerate(doc_chunks):
            elements.append(chunk.replace("\n", "").replace("'", ""))

        taille_batch = 500
        range_chunks = range(0, len(elements), taille_batch)
        progress_bar_chunks = ProgressBar(total=len(range_chunks))
        j = 0
        for i in range_chunks:
            doc_tokens += np.sum(self.data_manager.add_str_batch_elements(
                    elements=elements[i:i + taille_batch],
                    docs_name=name_docs[i : i + taille_batch],
                    path_docs=path_docs[i : i + taille_batch],
                    display_message=False
            ))
            progress_bar_chunks.update(j)
            j+=1
        progress_bar_chunks.clear()
        return doc_tokens

    def __serial_indexation__(self, doc_chunks, name_docs, path_docs):
        """
        Adds a batch of chunks from doc_chunks to the indexation verctorbase
        Args:
            doc_chunks (list[str]) : Chunks to be indexed
            name_docs (list[str]) : Name of docs each chunk is from

        Returns
            None
        """
        doc_tokens = 0
        for k, chunk in enumerate(doc_chunks):
                doc_tokens += self.data_manager.add_str_elements(
                    elements=[chunk.replace("\n", "").replace("'", "")],
                    docs_name=[name_docs[k]],
                    path_docs=[path_docs[k]],
                    display_message=False,
                )   
        return doc_tokens

    def run_pipeline(
        self, batch: bool = True, size_limit : int = 16000
    ) -> None:
        """
        Split texts from self.data_path, embed them and save them in a vector base.

        Args:
            size_limit (int) : Document size limit in number of characters, if exceeding document will be chunked in subdocuments of size size_limit
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
            embedding_tokens = 0
            input_tokens = 0
            output_tokens = 0
            doc = DocumentText(path=path_doc,
                               doc_index=i,
                               config_server={"data_preprocessing" : "pdf_text_extraction"},
                               splitter=self.splitter)
            doc_content = doc.content
            size_limit_doc = []
            left = 0
            #Découpage des documents en chunks de 8000 caractères ~5 pages
            while left + size_limit < len(doc_content):
                size_limit_doc.append(doc_content[left:left+size_limit])
                left += size_limit
            size_limit_doc.append(doc_content[left:])
            splitter = self.splitter
            # Chunkage de tout ces "petits documents"

            progress_bar_doc = ProgressBar(total=len(size_limit_doc))
            for j, little_doc in enumerate(size_limit_doc):
                doc_chunks = []
                name_docs = []

                chunked_docs = splitter.split_text(little_doc)
                for k in range(len(chunked_docs)):
                    doc_chunks.append(Chunk(text=chunked_docs[k], document=path_doc+f"_{j}", id=k))
                    name_docs.append(str(Path(path_doc).name)+f"_{j}")
                try:
                    name_docs = [str(Path(path_doc).name) for i in range(len(doc_chunks))]
                    path_docs = [str(Path(path_doc).parent) for i in range(len(doc_chunks))]
                    if batch:
                        chunk_with_context = run_batch_contextual(agent=self.agent,
                                                                        doc_chunks=doc_chunks,
                                                                        doc_content=little_doc,
                                                                        language=self.language)
                        chunk_with_context = chunk_with_context
                        input_tokens += chunk_with_context["nb_input_tokens"]
                        output_tokens += chunk_with_context["nb_output_tokens"]
                        embedding_tokens += self.__batch_indexation__(doc_chunks=chunk_with_context["texts"],
                                                        name_docs=name_docs, path_docs=path_docs
                                )

                    else:
                        chunk_with_context = run_serial_contextual(agent=self.agent,
                                                                        doc_chunks=doc_chunks,
                                                                        doc_content=doc_content,
                                                                        language=self.language)
                        chunk_with_context = chunk_with_context
                        input_tokens += chunk_with_context["nb_input_tokens"]
                        output_tokens += chunk_with_context["nb_output_tokens"]
                        embedding_tokens += self.__serial_indexation__(
                                    doc_chunks=chunk_with_context["texts"],
                                    name_docs=name_docs, path_docs=path_docs
                                )
                except:
                    None
                progress_bar_doc.update(j, text="Indexing {}, {}% done".format(path_doc,
                                                                                np.round((j/len(size_limit_doc)*100), 2)))
            new_doc = Document(name=str(Path(path_doc).name),
                               path=str(Path(path_doc)),
                               embedding_tokens=int(embedding_tokens),
                               input_tokens=int(input_tokens),
                               output_tokens=int(output_tokens))
            self.data_manager.add_instance(new_doc,
                                               path=str(Path(path_doc).parent))

            progress_bar.update(i)
        progress_bar.clear()