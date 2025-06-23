from ...database.rag_classes import DocumentText
from ...database.rag_classes import Document,Chunk
from ...utils.splitter import get_splitter
from .contextual import run_batch_contextual, run_serial_contextual
from tqdm.auto import tqdm
import os
from ...utils.progress import ProgressBar
import numpy as np
from .prompts import prompts


class ContextualRetrievalIndexation:
    "A class that handles the whole indexation process for the Contextual retrieval rag"

    def __init__(self, data_path: str, db,
                  vb, language: str, agent,
                  embedding_model :str,
                  type_text_splitter = "TextSplitter"
                  
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
        if data_path[-1] != "/":
            data_path += "/"
        self.data_path = data_path
        self.db = db
        self.vb = vb
        self.language = language
        self.agent = agent
        self.prompts = prompts[language]
        self.splitter = get_splitter(type_text_splitter=type_text_splitter,
                                     agent=self.agent, 
                                     embedding_model=embedding_model)
        self.input_tokens = 0
        self.output_tokens = 0

    def __batch_indexation__(self, doc_chunks: list[str], name_docs: list[str]):
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

        taille_batch = 50
        for i in range(0, len(elements), taille_batch):
            doc_tokens += np.sum(self.vb.add_str_batch_elements(
                    elements=elements[i:i + taille_batch],
                    docs_name=name_docs[i:i + taille_batch], 
                    display_message=False
            ))
        return doc_tokens

    def __serial_indexation__(self, doc_chunks, name_docs):
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
                doc_tokens += self.vb.add_str_elements(
                    elements=[chunk.replace("\n", "").replace("'", "")],
                    docs_name=[name_docs[k]],
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
        docs_already_processed = [res[0] for res in self.db.query(Document.name).all()]
        docs_to_process = [
            doc
            for doc in os.listdir(self.data_path)
            if doc not in docs_already_processed
        ]
        self.vb.create_collection()
        #with tqdm(docs_to_process) as progress_bar:
        progress_bar = ProgressBar(total=len(docs_to_process))
        for i, name_doc in enumerate(docs_to_process):
            embedding_tokens = 0
            input_tokens = 0
            output_tokens = 0
            doc = DocumentText(path=self.data_path + name_doc,
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
                    doc_chunks.append(Chunk(text=chunked_docs[k], document=name_doc+f"_{j}", id=k))
                    name_docs.append(name_doc+f"_{j}")
                try:
                    if batch:
                        chunk_with_context = run_batch_contextual(agent=self.agent,
                                                                        doc_chunks=doc_chunks,
                                                                        doc_content=little_doc,
                                                                        language=self.language)
                        chunk_with_context = chunk_with_context
                        input_tokens += chunk_with_context["nb_input_tokens"]
                        output_tokens += chunk_with_context["nb_output_tokens"]
                        embedding_tokens += self.__batch_indexation__(doc_chunks=chunk_with_context["texts"],
                                                        name_docs=name_docs
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
                                    name_docs=name_docs
                                )
                except:
                    None
                progress_bar_doc.update(j, text="Indexing {}, {}% done".format(name_doc,
                                                                                np.round((j/len(size_limit_doc)*100), 2)))
            new_doc = Document(name=name_doc, 
                               embedding_tokens=int(embedding_tokens),
                                input_tokens=int(input_tokens), output_tokens=int(output_tokens))
            self.db.add_instance(new_doc)

            progress_bar.update(i)