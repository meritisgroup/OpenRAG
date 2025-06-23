from ...utils.splitter import get_splitter
from ..graph_rag.extract_entities import DocumentText
from ...database.rag_classes import Document
from tqdm.auto import tqdm
import os
import numpy as np

class NaiveRagIndexation:
    def __init__(
        self,
        data_path: str,
        db,
        vb,
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
        if data_path[-1] != "/":
            data_path += "/"

        self.data_path = data_path

        self.db = db
        self.vb = vb
        self.agent = agent
        self.embedding_model = embedding_model

        self.splitter = get_splitter(type_text_splitter=type_text_splitter,
                                     agent=self.agent,
                                     embedding_model=self.embedding_model)

    def __batch_indexation__(self, doc_chunks, name_docs):
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
        for i in range(0, len(elements), taille_batch):
            tokens += np.sum(self.vb.add_str_batch_elements(
                    elements=elements[i:i + taille_batch],
                    docs_name=name_docs[i:i + taille_batch], 
                    display_message=False
            ))
        return tokens

    def __serial_indexation__(self, doc_chunks, name_docs):
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
                tokens += self.vb.add_str_elements(
                    elements=[chunk.text.replace("\n", "").replace("'", "")],
                    docs_name=[name_docs[k]],
                    display_message=False,
                )
            except Exception:
                None
        return tokens

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
        docs_already_processed = [res[0] for res in self.db.query(Document.name).all()]
        docs_to_process = [
            doc
            for doc in os.listdir(self.data_path)
            if doc not in docs_already_processed
        ]

        self.vb.create_collection()
        with tqdm(docs_to_process) as progress_bar:

            for i, name_doc in enumerate(progress_bar):
                doc_indexation_tokens = 0
                progress_bar.set_description(f"Embbeding chunks - {name_doc}")
                doc = DocumentText(
                    path=self.data_path + name_doc, splitter=self.splitter
                )
                doc_chunks = doc.chunks(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                name_docs = [name_doc for i in range(len(doc_chunks))]
                if True:
                    if batch:
                        doc_indexation_tokens += self.__batch_indexation__(
                            doc_chunks=doc_chunks, name_docs=name_docs
                        )

                    else:
                        doc_indexation_tokens += self.__serial_indexation__(
                            doc_chunks=doc_chunks, name_docs=name_docs
                        )
      

                # print("Failed indexing: {}".format(name_doc))
                if i == len(progress_bar) - 1:
                    progress_bar.set_description("Embbeding chunks - ✅")

                new_doc = Document(name=name_doc,embedding_tokens=int(doc_indexation_tokens), input_tokens=0, output_tokens=0)
                self.db.add_instance(new_doc)
                # print(b-a, c-b, time.time()-c)
