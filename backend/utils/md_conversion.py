from pathlib import Path
import logging
from io import BytesIO
import base64
import fitz
from PIL import Image
import concurrent.futures
from .threading_utils import get_executor_threads


class VlmConverter:

    def __init__(self, config_server, agent):
        self.config_server = config_server
        self.agent = agent


    def convert(self, input_file: str, image_description: bool = True, max_workers: int = 10) -> str | None:
        input_path = Path(input_file)
        doc = fitz.open(input_path)
        images = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()

        prompts = ["""Extrait le texte du document ci-dessus comme si tu le lisais naturellement.
Renvoie le document sous format Markdown.
Renvoie les tableaux au format HTML.
Renvoie les équations sous forme de représentation LaTeX.
S’il y a une image dans le document, ajoute une description précise de l’image à l’intérieur de la balise <img></img> et insère également la légende de l’image dans cette même balise.
Écris les descriptions des images dans la langue du document.
Les filigranes doivent être entourés de balises, par exemple : <watermark>OFFICIAL COPY</watermark>.
Les numéros de page doivent également être entourés de balises, par exemple : <page_number>14</page_number> ou <page_number>9/22</page_number>.
Privilégie l’utilisation des symboles ☐ et ☑ pour les cases à cocher.""" for i in range(len(images))]
        results = self.agent.predict_images(prompts=prompts,
                                            model=self.config_server["model_for_image"],
                                            max_workers=max_workers,
                                            images=images,
                                            temperature=0.5)
        markdown = "".join(results)
        return markdown
