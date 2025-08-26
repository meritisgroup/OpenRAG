import ast
from ...database.rag_classes import Document
from ..graph_rag.extract_entities import DocumentText
from ...utils.splitter import get_splitter
from .prompts import prompts
import numpy as np

from tqdm.auto import tqdm
from ...utils.progress import ProgressBar
import os


class QbRagIndexation:
    def __init__(
        self,
        data_path: str,
        vb,
        db,
        agent,
        embedding_model,
        language: str = "EN",
        type_text_splitter="TextSplitter",
    ):

        if data_path[-1] != "/":
            data_path += "/"

        self.data_path = data_path

        self.db = db
        self.db_name = db.name[:-3]
        self.vb = vb
        self.agent = agent
        self.embedding_model = embedding_model
        self.language = language
        self.prompts = prompts[language]
        self.splitter = get_splitter(
            type_text_splitter=type_text_splitter,
            agent=self.agent,
            embedding_model=self.embedding_model,
        )

    def generates_questions(self, chunks: list[str], temperature=1, text_to_show="Generate questions"):
        prompts = []
        system_prompts = []
        for chunk in chunks:
            prompt = self.prompts["qb_prompt"]["QUERY_TEMPLATE"].format(query=chunk)
            system_prompt = self.prompts["qb_prompt"]["SYSTEM_PROMPT"]
            prompts.append(prompt)
            system_prompts.append(system_prompt)
        
        tokens = 0
        taille_batch = 100
        outputs = None
        data_range = range(0, len(prompts), taille_batch)
        progress_bar = ProgressBar(total=len(data_range))
        k = 0
        for i in data_range:
            results = self.agent.multiple_predict(prompts=prompts[i:i + taille_batch],
                                            system_prompts=system_prompts[i:i + taille_batch])
            k+=1
            progress_bar.update(k-1, text=text_to_show+" {}".format(np.round((k/len(data_range))*100,2)))
            if outputs is None:
                outputs = results
            else:
                outputs["texts"] = outputs["texts"] + results["texts"]
                outputs["nb_input_tokens"]+=np.sum(results["nb_input_tokens"])
                outputs["nb_output_tokens"]+=np.sum(results["nb_output_tokens"])
                outputs["impacts"][0]+=results["impacts"][0]
                outputs["impacts"][1]+=results["impacts"][1]
                outputs["energy"][0]+=results["energy"][0]
                outputs["energy"][1]+=results["energy"][1]

        return outputs

    def __batch_indexation__(self, doc_chunks, name_docs):
        """
        Adds a batch of chunks from doc_chunks to the indexation verctorbase
        Args:
            doc_chunks (list[str]) : Chunks to be indexed
            name_docs (list[str]) : Name of docs each chunk is from

        Returns
            None
        """
        embedding_tokens, input_tokens, output_tokens = 0, 0, 0
        elements = []
        for k, chunk in enumerate(doc_chunks):
            elements.append(chunk.text.replace("\n", "").replace("'", ""))
        list_questions = self.generates_questions(chunks=elements,
                                                  text_to_show="Generate questions")

        input_tokens += np.sum(list_questions["nb_input_tokens"])
        output_tokens += np.sum(list_questions["nb_output_tokens"])

        final_name_docs = []
        metadatas = []
        questions_docs = []
        success = False
        nb_try = 1
        for i in range(1):
            elements_to_retry = []
            name_docs_to_retry = []
            for k, questions in enumerate(list_questions["texts"]):
                try:
                    questions1 = ast.literal_eval(questions)
                    for i in range(len(questions1)):
                        metadatas.append({"chunk_text": elements[k]})
                        final_name_docs.append(name_docs[k])
                        questions_docs.append(questions1[i])
                except Exception:
                    elements_to_retry.append(elements[k])
                    name_docs_to_retry.append(name_docs[k])

            if len(elements_to_retry)>0:
                list_questions = self.generates_questions(chunks=elements_to_retry,
                                                        temperature=1,
                                                        text_to_show="Regenerate fail questions")
                elements = elements_to_retry
                name_docs = name_docs_to_retry
                input_tokens += np.sum(list_questions["nb_input_tokens"])
                output_tokens += np.sum(list_questions["nb_output_tokens"])

        taille_batch = 500
        for i in range(0, len(elements), taille_batch):
            embedding_tokens += np.sum(self.vb.add_str_batch_elements(collection_name=self.db_name,
                                                                      elements=questions_docs[i:i + taille_batch],
                                                                      display_message=False,
                                                                      metadata=metadatas[i:i + taille_batch],
                                                                      docs_name=final_name_docs[i:i + taille_batch]))
        return embedding_tokens, input_tokens, output_tokens

    def __serial_indexation__(self, doc_chunks, name_docs):
        """
        Adds a batch of chunks from doc_chunks to the indexation verctorbase
        Args:
            doc_chunks (list[str]) : Chunks to be indexed
            name_docs (list[str]) : Name of docs each chunk is from

        Returns
            None
        """
        embedding_tokens, input_tokens, output_tokens = 0, 0, 0
        for k, chunk in enumerate(doc_chunks):
            questions = self.generates_questions(chunks=[chunk.text])
            input_tokens += np.sum(questions["nb_input_tokens"])
            output_tokens += np.sum(questions["nb_output_tokens"])

            for question_str in questions["texts"]:
                success = False
                nb_try = 1
                while not success:
                    try:
                        questions1 = ast.literal_eval(questions)
                        success = True
                    except Exception:
                        questions = self.generates_questions(
                            chunks=[chunk.text], temperature=1
                        )["texts"][0]
                        input_tokens += np.sum(questions["nb_input_tokens"])
                        output_tokens += np.sum(questions["nb_output_tokens"])
                        success = False
                        nb_try+=1
                        if nb_try>10:
                            break;
                try:
                    if success:
                        for question in questions1:
                            embedding_tokens += self.vb.add_str_elements(
                                collection_name=self.db_name,
                                elements=[question],
                                display_message=False,
                                metadata=[{"chunk_text": chunk.text}],
                                docs_name=[name_docs[k]],
                            )
                except Exception:
                    None
        return embedding_tokens, input_tokens, output_tokens

    def run_pipeline(self, chunk_size, batch: bool = True):
        add_fields = [
            {
                "field_name": "chunk_text",
                "data": {
                    "field_name": "chunk_text",
                    "datatype": "str",
                    "max_length": 5001,
                },
            }
        ]
        self.vb.create_collection(
            name=self.db_name, add_fields=add_fields
        )
        docs_already_processed = [res[0] for res in self.db.query(Document.name).all()]
        docs_to_process = [
            doc
            for doc in os.listdir(self.data_path)
            if doc not in docs_already_processed
        ]
        progress_bar = ProgressBar(docs_to_process)
        for i, name_doc in enumerate(progress_bar.iterable):
            embedding_tokens, input_tokens, output_tokens = 0, 0, 0
            progress_bar.update(
                i - 1, f"Creating question cache for document : {name_doc}"
            )
            doc = DocumentText(path=self.data_path + name_doc,config_server={"data_preprocessing" : "pdf_text_extraction"}, splitter=self.splitter)
            doc_chunks = doc.chunks(chunk_size=chunk_size, chunk_overlap=False)
            name_docs = [name_doc for i in range(len(doc_chunks))]
            if batch:
                embedding_t, input_t, output_t = self.__batch_indexation__(
                    doc_chunks=doc_chunks, name_docs=name_docs
                )
                embedding_tokens += embedding_t
                input_tokens += input_t
                output_tokens += output_t
            else:
                embedding_t, input_t, output_t = self.__serial_indexation__(
                    doc_chunks=doc_chunks, name_docs=name_docs
                )
                embedding_tokens += embedding_t
                input_tokens += input_t
                output_tokens += output_t 
            new_doc = Document(
                name=name_doc,
                embedding_tokens=int(embedding_tokens),
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
            )
            self.db.add_instance(new_doc)
        progress_bar.clear()
