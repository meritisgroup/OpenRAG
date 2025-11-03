from typing import List
import os
import json
import re
from pathlib import Path
from ..utils.open_doc import Opener
from ..utils.open_markdown_doc import MarkdownOpener
from ..utils.splitter import TextSplitter
from ..utils.base_classes import Splitter
from ..utils.splitter import MarkdownHeaderTextSplitter
from ..utils.splitter import get_splitter
from .rag_classes import Chunk, Document

import secrets
import string

# Alphabet base62 (URL/file-safe)
_ALPHABET = string.ascii_letters + string.digits  # 26+26+10 = 62


def make_chunk_id() -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(15))


def save_chunks_to_jsonl(
    doc_id, chunks: List[Chunk], jsonl_path: str = "chunks_output.jsonl"
):

    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for c in chunks:
            record = {col.name: getattr(c, col.name) for col in c.__table__.columns}
            record["doc_id"] = doc_id
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return None


class DocumentText:
    def __init__(
        self,
        doc_index,
        path: str,
        config_server: dict,
        agent,
        splitter: Splitter = TextSplitter(),
        reset_preprocess = False
    ):

        self.data_preprocessing = config_server["data_preprocessing"]
        self.name_with_extension = Path(path).name
        self.config_server = config_server
        self.agent = agent
        self.doc_index = doc_index
        self.path = path
        self.reset_preprocess = reset_preprocess

        try:
            if self.data_preprocessing == "md_with_images":
                if self.reset_preprocess or not self.load_content__("md"):
                    self.content = MarkdownOpener(config_server=self.config_server,
                                                  agent=self.agent,
                                                  image_description=True).open_doc(path_file=path)
                    self.save_content__(format="md")
            elif self.data_preprocessing == "md_without_images":
                if self.reset_preprocess or not self.load_content__("md"):
                    self.content = MarkdownOpener(config_server=self.config_server,
                                                  agent=self.agent,
                                                  image_description=False).open_doc(path_file=path)
                    self.save_content__(format="md")
            else:
                if not self.load_content__("txt"):
                    self.content = Opener(save=True).open_doc(path)
                    self.save_content__(format="txt")
            
        except Exception as e:
            self.content = ""
            print(f'Error "{e}" while trying to open doc {self.name_with_extension}')

        self.name = ".".join(self.name_with_extension.split(".")[:-1])
        self.extension = "." + self.name_with_extension.split(".")[-1]
        
        self.text_splitter = splitter

    
    def load_content__(self, format: str):
        file = os.path.join(Path(self.path).parent,
                            self.data_preprocessing,
                            Path(self.name_with_extension).with_suffix("."+format).name)
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                self.content = f.read()  
            return True
        else:
            return False

    def save_content__(self, format: str):
        file_save = os.path.join(Path(self.path).parent, self.data_preprocessing)
        os.makedirs(file_save, exist_ok=True)
        file_save = os.path.join(file_save,
                                Path(self.name_with_extension).with_suffix("."+format).name)
        with open(file_save, "w", encoding="utf-8") as f:
            f.write(self.content)


    def get_content(self):
        return self.content
    
    def chunks(self, chunk_size: int = 1024, chunk_overlap: bool = True) -> list[Chunk]:

        results = []
        chunk_id = 1
        """
        # Fallback for non-markdown inputs
        """
        chunks = self.text_splitter.split_text(text=self.content, 
                                               chunk_size=chunk_size, 
                                               overlap=chunk_overlap)

            for k, text in enumerate(texts):
                results.append(
                    Chunk(text=text, 
                          document=self.name_with_extension, 
                          position_in_doc=k + 1,
                          id=make_chunk_id())
                )

        # save chunks
        #save_chunks_to_jsonl(self.doc_index, results)
        return results

    def convert_in_base(self) -> Document:
        return Document(
            name=self.name_with_extension,
            path=str(self.path),
            embedding_tokens=0,
            input_tokens=0,
            output_tokens=0,
        )
