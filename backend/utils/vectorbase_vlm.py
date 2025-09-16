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


