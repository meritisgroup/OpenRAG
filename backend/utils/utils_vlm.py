import requests
import io
import fitz
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import base64
import numpy as np
import os
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from .agent import Agent


def make_request_vlm_ollama(url, image, model, prompt):
    url = url + "/api/generate"
    image_io = io.BytesIO()
    image.save(image_io, format="png")
    image_io.seek(0)

    base64_image = base64.b64encode(image_io.getvalue()).decode("utf-8")

    data = {"model": model, "prompt": prompt, "images": [base64_image], "stream": False}

    response = requests.post(url, json=data)

    return response.json()["response"]


def run_generation_vlm_hf(queries, images, model):
    answers = []
    if model["model_name"] == "openbmb/MiniCPM-V-2_6":
        for i in range(len(queries)):
            msgs = [{"role": "user", "content": images[i] + [queries[i]]}]
            answer = model["model"].chat(
                image=None,
                msgs=msgs,
                tokenizer=model["tokenizer"],
                processor=model["processor"],
            )
            answers.append(answer)
    elif "llava" in model["model_name"]:
        for i in range(len(queries)):
            answer_temp = []
            answer = ""
            for j in range(len(images[i])):
                answer_temp.append(
                    make_request_vlm_ollama(
                        url=model["url"],
                        image=images[i][j],
                        model=model["model_name"],
                        prompt=queries[i],
                    )
                )

                answer = answer + "\n\n" + answer_temp[-1]
            answers.append(answer)
    return answers


def load_vlm_model(model_params, device="cpu"):

    if model_params["type"] == "HF":
        HF_TOKEN = ""
        model_tokenizer = AutoTokenizer.from_pretrained(
            model_params["model_name"],
            attn_implementation="sdpa",
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        model = AutoModel.from_pretrained(
            model_params["model_name"],
            trust_remote_code=True,
            attn_implementation="sdpa",
            token=HF_TOKEN,
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_params["model_name"], trust_remote_code=True, token=HF_TOKEN
        )
        model = model.to(device)
        return {
            "model": model,
            "tokenizer": model_tokenizer,
            "processor": processor,
            "model_name": model_params["model_name"],
        }
    else:
        data = {
            "agent": Agent(
                model=model_params["model_name"],
                key_or_url=model_params["url"],
                language=model_params["language"],
            )
        }
        for key in model_params.keys():
            data[key] = model_params[key]
        return data


def load_xlsx_images(xlsx_file):
    path = xlsx_file.split(":")[0]
    df = pd.read_excel(path)

    dfi.export(df, "tableau.png")

    image = Image.open("tableau.png")
    os.remove("tableau.png")
    return image, False


def load_txt_images(txt_file):
    path = txt_file.split(":")[0]
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    font = ImageFont.load_default()
    line_height = 20
    image_width = 800
    image_height = line_height * len(lines) + 20

    image = Image.new("RGB", (image_width, image_height), color="white")
    draw = ImageDraw.Draw(image)
    y = 10
    for line in lines:
        draw.text((10, y), line.strip(), fill="black", font=font)
        y += line_height

    return image, False


def pdf_to_images(pdf_path):
    if type(pdf_path) is str:
        pdf_path = [pdf_path]

    images = []
    for path in pdf_path:
        pdf_document = fitz.open(path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=200)

            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            images.append(image)
    return images


def load_page_pdf_images(pdf_path, format="PIL"):
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
    if format == "numpy":
        image = np.array(image)
    return image, is_empty


def load_image(path):
    path = path.split(":")[0]
    img = Image.open(path)
    return img


def set_vllm_HF_key(url, key):
    data = {"key": key}
    url = url + "/setHFkey"
    requests.post(url, json=data).json()


def load_element(path_element):
    if ".pdf" in path_element:
        element, is_empty = load_page_pdf_images(pdf_path=path_element)
        """
        elif ".xlsx" in path_element:
            element, is_empty = load_xlsx_images(xlsx_file=path_element)
        elif ".txt" in path_element:
            element, is_empty = load_txt_images(txt_file=path_element)
        elif ".docx" in path_element:
        """
    elif (
        ".xlsx" in path_element
        or ".txt" in path_element
        or ".docx" in path_element
        or ".pptx" in path_element
    ):
        return None, True

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
