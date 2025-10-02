import ast
from ...database.rag_classes import Document, Chunk_query
from ...database.data_extraction import DocumentText
from ...utils.splitter import get_splitter
from .prompts import prompts
import numpy as np
from pathlib import Path
import json

from tqdm.auto import tqdm
from ...utils.progress import ProgressBar
import os
from pydantic import BaseModel
from typing import List


class Questions_Model(BaseModel):
    questions: List[str]

class QbRagIndexation:
    def __init__(
        self,
        data_manager,
        agent,
        embedding_model,
        language: str = "EN",
        type_text_splitter="TextSplitter",
    ):
        
        self.data_manager = data_manager
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
        taille_batch = 1000
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


    def __indexation__(self, doc_chunks, path_docs):
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

        final_path_docs = []
        metadatas = []
        final_chunks = []
        questions_docs = []
        success = False
        nb_try = 1
        for i in range(1):
            elements_to_retry = []
            for k, questions in enumerate(list_questions["texts"]):
                try:
                    questions1 = json.loads(questions)
                    for i in range(len(questions1)):
                        metadatas.append({"chunk_text": elements[k]})
                        final_path_docs.append(path_docs[k])
                        questions_docs.append(questions1[i])
                        new_chunk = Chunk_query.from_chunk(chunk=doc_chunks[k],
                                                           query=questions1[i])
                        final_chunks.append(new_chunk)
                except Exception:
                    elements_to_retry.append(elements[k])
            
            if len(elements_to_retry)>0:
                list_questions = self.generates_questions(chunks=elements_to_retry,
                                                          temperature=1,
                                                          text_to_show="Regenerate fail questions")
                elements = elements_to_retry
                input_tokens += np.sum(list_questions["nb_input_tokens"])
                output_tokens += np.sum(list_questions["nb_output_tokens"])
            

        taille_batch = 1000
        for i in range(0, len(final_chunks), taille_batch):
            embedding_tokens += np.sum(self.data_manager.add_str_batch_elements(display_message=False,
                                                                                chunks = final_chunks[i:i + taille_batch],
                                                                                path_docs=final_path_docs[i:i + taille_batch]))
        return embedding_tokens, input_tokens, output_tokens


    def run_pipeline(self, 
                     config_server,
                     chunk_size, 
                     chunk_overlap: bool = True,
                     reset_preprocess: bool = False):
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
        self.data_manager.create_collection(add_fields=add_fields)
        docs_already_processed = [res[0] for res in self.data_manager.query(Document.path)]
        to_process_norm = [Path(p).resolve().as_posix() for p in self.data_manager.get_list_path_documents()]
        docs_already_norm = [Path(p).resolve().as_posix() for p in docs_already_processed]

        docs_to_process = [
            doc
            for doc in to_process_norm
            if doc not in docs_already_norm
        ]

        self.data_manager.create_collection()
        progress_bar = ProgressBar(docs_to_process)
        for i, path_doc in enumerate(progress_bar.iterable):
            embedding_tokens, input_tokens, output_tokens = 0, 0, 0
            progress_bar.update(
                i - 1, f"Creating question cache for document : {path_doc}"
            )
            doc = DocumentText(path=path_doc,
                               doc_index=i,
                               config_server=config_server,
                               splitter=self.splitter,
                               reset_preprocess=reset_preprocess)
            doc_chunks = doc.chunks(chunk_size=chunk_size, 
                                    chunk_overlap=chunk_overlap)
            name_docs = [str(Path(path_doc).name) for i in range(len(doc_chunks))]
            path_docs = [str(Path(path_doc).parent) for i in range(len(doc_chunks))]

            embedding_t, input_t, output_t = self.__indexation__(doc_chunks=doc_chunks,
                                                                 path_docs=path_docs)
                
            embedding_tokens += embedding_t
            input_tokens += input_t
            output_tokens += output_t

            new_doc = Document(name=str(Path(path_doc).name),
                               path=str(Path(path_doc)),
                               embedding_tokens=int(embedding_tokens),
                               input_tokens=int(input_tokens),
                               output_tokens=int(output_tokens))
            self.data_manager.add_instance(instance=new_doc,
                                           path=str(Path(path_doc).parent))
        progress_bar.clear()
