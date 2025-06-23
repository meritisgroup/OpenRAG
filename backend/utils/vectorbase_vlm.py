from pymilvus import MilvusClient, DataType
import torch
from typing import Union
import numpy as np
import fitz
import torch.nn.functional as F
from PIL import Image


def pdf_to_images(pdf_path):
    if type(pdf_path) is str:
        pdf_path = [pdf_path]

    images = []
    for path in pdf_path:
        pdf_document = fitz.open(path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()

            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            images.append(image)
    return images


def load_page_pdf_images(pdf_path):
    page = int(pdf_path.split(":")[1])
    path = pdf_path.split(":")[0]
    pdf_document = fitz.open(path)
    page = pdf_document.load_page(page)

    text = page.get_text("text")
    images = page.get_images(full=True)
    if text.strip() or images:
        is_empty = False
    else:
        is_empty = True

    pix = page.get_pixmap(dpi=200)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image, is_empty


def load_image(path):
    path = path.split(":")[0]
    img = Image.open(path)
    return img


def load_element(path_element):
    if ".pdf" in path_element:
        element, is_empty = load_page_pdf_images(pdf_path=path_element)
    else:
        element = load_image(path=path_element)
        if element is None:
            is_empty = True
        else:
            is_empty = False
    return element, is_empty


def get_embedding_vlm(datas, model, tokenizer_model):
    inputs_image = []
    inputs_text = []
    for i in range(len(datas)):
        if type(datas[i]) is str:
            inputs_text.append(datas[i])
            inputs_image.append(None)
        else:
            inputs_image.append(datas[i])
            inputs_text.append("Can you describe and extract the information?")

    inputs = {"text": inputs_text, "image": inputs_image, "tokenizer": tokenizer_model}

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    attention_mask = outputs.attention_mask

    attention_mask_cumulative = attention_mask * attention_mask.cumsum(dim=1)
    weighted_sum = torch.sum(
        last_hidden_state * attention_mask_cumulative.unsqueeze(-1).float(), dim=1
    )
    sum_of_weights = attention_mask_cumulative.sum(dim=1, keepdim=True).float()
    weighted_mean = weighted_sum / sum_of_weights
    embeddings = F.normalize(weighted_mean, p=2, dim=1).detach().cpu().numpy()
    return embeddings


class VectorBaseVlm:
    def __init__(
        self,
        vb_name: str,
        path: str,
        embedding_model,
        tokenizer_model,
        erase_existing=True,
    ):

        if len(vb_name) < 3 or vb_name[-3:] != ".db":
            vb_name += ".db"

        if path[-1] != "/":
            path += "/"

        name = path + vb_name

        self.client = MilvusClient(name)
        self.embedding_model = embedding_model
        self.tokenizer_model = tokenizer_model
        self.dimension = len(
            get_embedding_vlm(
                datas=["test"],
                model=self.embedding_model,
                tokenizer_model=self.tokenizer_model,
            )[0]
        )

    def get_current_doc_id(self, collection_name):
        if not self.client.has_collection(collection_name):
            return 0
        else:
            result = self.client.query(
                collection_name=collection_name,
                filter="id >= 0",
                output_fields=["doc_id", "path"],
            )
            doc_id = [element["doc_id"] for element in result]
            if len(doc_id) > 0:
                return np.max(doc_id)
            return 0

    def create_collection(self, name: str) -> None:
        if not self.client.has_collection(name):
            schema = self.client.create_schema(auto_id=True, enable_dynamic_fields=True)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT16_VECTOR,
                dim=self.dimension,
            )
            schema.add_field(field_name="doc_id", datatype=DataType.INT64)
            schema.add_field(
                field_name="path", datatype=DataType.VARCHAR, max_length=65535
            )

            self.client.create_collection(collection_name=name, schema=schema)

            self.client.release_collection(collection_name=name)

            self.client.drop_index(collection_name=name, index_name="vector")
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_name="vector_index",
                index_type="FLAT",
                metric_type="IP",
                params={
                    "M": 16,
                    "efConstruction": 500,
                },
            )

            self.client.create_index(
                collection_name=name, index_params=index_params, sync=True
            )
        else:
            print(f'The collection "{name}" is already existing')

    def construct_test(self, element_tags: dict) -> str:
        filter = ""
        for key in element_tags.keys():
            filter += f"{key} == '{element_tags[key]}' and "

        return filter[:-5]

    def add_str_elements(
        self,
        collection_name: str,
        elements: list[str],
        doc_id: int,
        metadata: dict = None,
        display_message: bool = True,
    ) -> None:

        if not self.client.has_collection(collection_name):
            self.create_collection(collection_name)

        if metadata is not None:
            filtered_elements = [
                element
                for element in elements
                if len(
                    self.client.query(
                        collection_name=collection_name,
                        filter=f"path == '{element}' and {self.construct_test(metadata[element])}",
                        output_fields=["path"],
                    )
                )
                == 0
            ]

        else:
            filtered_elements = [
                element
                for element in elements
                if len(
                    self.client.query(
                        collection_name=collection_name,
                        filter=f"path == '{element}'",
                        output_fields=["path"],
                    )
                )
                == 0
            ]

        data = []
        if filtered_elements != []:
            images = []
            for i in range(len(filtered_elements)):
                image, is_empty = load_element(filtered_elements[i])
                if not is_empty:
                    images.append(image)
            if len(images) > 0:
                embeddings = get_embedding_vlm(
                    datas=images,
                    model=self.embedding_model,
                    tokenizer_model=self.tokenizer_model,
                ).astype(np.float16)

                if metadata is not None:
                    data = [
                        {"doc_id": doc_id, "vector": embeddings[k], "path": element}
                        | metadata[element]
                        for k, element in enumerate(filtered_elements)
                    ]

                else:
                    data = [
                        {"doc_id": doc_id, "vector": embeddings[k], "path": element}
                        for k, element in enumerate(filtered_elements)
                    ]

        if data != []:
            self.client.insert(collection_name=collection_name, data=data)
            if display_message:
                print(
                    f"{len(data)} elements have been successfuly added in the vector base"
                )
        else:
            if display_message:
                print(
                    f"All the elements already were in the collection {collection_name}"
                )

    def k_search(
        self,
        queries: Union[str, list[str]],
        collection_name: str,
        k: int,
        output_fields: list[str] = ["path"],
        filters: dict = None,
    ):

        if type(queries) is str:
            queries = [queries]

        data = get_embedding_vlm(
            datas=queries,
            model=self.embedding_model,
            tokenizer_model=self.tokenizer_model,
        ).astype(np.float16)

        if filters is not None:
            res = self.client.search(
                collection_name=collection_name,
                data=data,
                filter=self.construct_test(filters),
                limit=k,
                output_fields=output_fields,
            )

        else:
            res = self.client.search(
                collection_name=collection_name,
                data=data,
                limit=k,
                output_fields=output_fields,
            )
        results = []
        for l in range(len(res)):
            result = []
            for i in range(np.min([len(res[l]), k])):
                result.append(res[l][i]["entity"])
            results.append(result)
        return results
