from typing import List
import os
import json
import re
from ..utils.open_doc import Opener
from ..utils.open_markdown_doc import MarkdownOpener
from ..utils.splitter import TextSplitter
from ..utils.base_classes import Splitter
from ..utils.splitter import MarkdownHeaderTextSplitter
from ..utils.splitter import get_splitter
from .rag_classes import Chunk, Document


def save_chunks_to_jsonl(
    doc_id, chunks: List["Chunk"], jsonl_path: str = "chunks_output.jsonl"
):

    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for c in chunks:
            record = {
                "name_doc": getattr(c, "document", None),
                "doc_id": doc_id,
                "chunk_id": getattr(c, "id", None),
                "chunk_content": str(getattr(c, "text", "")),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return None


def extract_images_and_tables_blocks(section: str) -> list[str]:
    """
    Découpe une section Markdown en blocs :
    - Blocs d'image (délimités par <IMAGE X --> ... <IMAGE X -->)
    - Blocs de tableau (au moins deux lignes commençant et finissant par '|')
    - Blocs de texte (tout le reste)
    """
    parts = []
    last_idx = 0

    # Step 1 : extract images blocs
    image_block_pattern = re.compile(
        r"(<IMAGE\s+\d+\s*-->\s*\n.*?\n<IMAGE\s+\d+\s*-->)", re.DOTALL
    )

    image_blocks = []
    for match in image_block_pattern.finditer(section):
        start, end = match.span()

        if start > last_idx:
            before = section[last_idx:start]
            image_blocks.append(("text", before))

        image_block = match.group(1)
        image_blocks.append(("image", image_block))

        last_idx = end

    if last_idx < len(section):
        image_blocks.append(("text", section[last_idx:]))

    # Step 2 : for each "text" bloc, detect tables inside
    table_pattern = re.compile(r"((?:^\|[^\n]*\|\s*\n?){2,})", re.MULTILINE)

    for block_type, content in image_blocks:
        if block_type == "image":
            parts.append(content.strip())  # we keep the image block as it is
        else:
            last_idx = 0
            for match in table_pattern.finditer(content):
                start, end = match.span()

                if start > last_idx:
                    before = content[last_idx:start].strip()
                    if before:
                        parts.append(before)

                table_block = match.group(1).strip()
                parts.append(table_block)

                last_idx = end

            if last_idx < len(content):
                after = content[last_idx:].strip()
                if after:
                    parts.append(after)

    return parts


class DocumentText:
    def __init__(
        self,
        doc_index,
        path: str,
        config_server: dict,
        splitter: Splitter = TextSplitter(),
    ):

        self.data_preprocessing = config_server["data_preprocessing"]
        self.name_with_extension = path.split("/")[-1]
        self.config_server = config_server
        self.doc_index = doc_index

        try:
            if self.data_preprocessing == "md_with_images":
                self.content = MarkdownOpener(
                    config_server=self.config_server
                ).open_doc(path_file=path)
            elif self.data_preprocessing == "md_without_images":
                self.content = MarkdownOpener(
                    config_server=self.config_server, image_description=False
                ).open_doc(path_file=path)
            else:
                self.content = Opener(save=False).open_doc(path)

        except Exception as e:
            self.content = ""
            print(f'Error "{e}" while trying to open doc {self.name_with_extension}')

        self.name = ".".join(self.name_with_extension.split(".")[:-1])
        self.extension = "." + self.name_with_extension.split(".")[-1]
        if (
            self.data_preprocessing == "md_with_images"
            or self.data_preprocessing == "md_without_images"
        ):
            self.text_splitter = TextSplitter()
        else:
            self.text_splitter = splitter

    def chunks(self, chunk_size: int = 500, chunk_overlap: bool = True) -> list[Chunk]:

        results = []
        chunk_id = 1

        if (
            self.data_preprocessing == "md_with_images"
            or self.data_preprocessing == "md_without_images"
        ):
            # Create the markdown splitter
            markdown_splitter = MarkdownHeaderTextSplitter(strip_headers=True)

            # Split the markdown text. The output is a dict : {{'page_content: str, 'metadata': {'Header 2' : str, 'Header 3' : str}}, {'page_content: str, 'metadata': {'Header 2' : str, 'Header 3' : str}}}
            md_sections = markdown_splitter.split_text(self.content)
            for section in md_sections:
                section_text = section["page_content"]  # page_content of a section
                section_headers = section["metadata"]

                # Concatenate headers of a section into a string which will be added to chunks for context
                headers_str = " ".join(
                    str(v) for v in section_headers.values() if v
                ).strip()

                parts = extract_images_and_tables_blocks(section_text)

                chunks = []
                for part in parts:
                    if re.match(r"^<IMAGE\s+\d+\s*-->", part):  # image description bloc
                        chunks.append(part)
                    elif re.match(r"^\s*\|.*\|\s*$", part, re.MULTILINE):  # table bloc
                        chunks.append(part)

                    else:
                        # Apply classic text splitter to non-table part
                        sub_chunks = self.text_splitter.split_text(
                            text=part, chunk_size=chunk_size, overlap=chunk_overlap
                        )

                        chunks.extend(sub_chunks)

                for chunk in chunks:
                    if headers_str:
                        final_text = headers_str + "\n" + chunk
                    else:
                        final_text = chunk
                    results.append(
                        Chunk(
                            text=final_text,
                            document=self.name_with_extension,
                            id=chunk_id,
                        )
                    )
                    chunk_id += 1

        else:
            """
            # Fallback for non-markdown inputs
            """
            chunks = self.text_splitter.split_text(
                text=self.content, chunk_size=chunk_size, overlap=chunk_overlap
            )

            for k, chunk in enumerate(chunks):
                results.append(
                    Chunk(text=chunk, document=self.name_with_extension, id=k + 1)
                )

        # save chunks
        save_chunks_to_jsonl(self.doc_index, results)
        """
        save_chunks_to_file(results)
        """
        return results

    def convert_in_base(self) -> Document:
        return Document(
            name=self.name_with_extension,
            embedding_tokens=0,
            input_tokens=0,
            output_tokens=0,
        )
