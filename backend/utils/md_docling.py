import base64
from typing import Optional
from pydantic import BaseModel
from pathlib import Path
from io import BytesIO
import logging
import numpy as np

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem, TextItem, GroupItem, DocItemLabel
from docling.exceptions import ConversionError
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

from ..utils.agent import get_Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class ImageClassification(BaseModel):
    category: str


class ImageDescription(BaseModel):
    description: str



classification_prompt = """
Tu dois classer l’image présentée dans l’un des types suivants :

[GRAPHIQUE, TABLEAU, DIAGRAMME, ORGANIGRAMME, CARTE GEOGRAPHIQUE, SCHEMA TECHNIQUE,
 SCAN DE DOCUMENT, PHOTO, LOGO, HISTOGRAMME]

Règles :
- Donne uniquement un seul type issu de la liste.
- Ne donne aucune explication ou justification.
- Utilise uniquement les termes de la liste ci-dessus, sans ajout.

Question :
Classifie cette image parmi les types d'image listés ci-dessus.
"""


def get_prompt(type_image, context):
    if type_image == "GRAPHIQUE":
        rules = """
        1. Type de graphique (barres, courbes, secteurs, etc.)
        2. Variables représentées (axes X et Y), leur échelle, leur unité de mesure
        3. Toutes les valeurs numériques dans un tableau markdown
        4. Tendances, points remarquables, évolution... le plus précisèment possible
        5. Source des données si visible"""
    elif type_image == "TABLEAU":
        rules = """
    1. Structure du tableau (nombre de lignes/colonnes)
    2. En-têtes de colonnes et lignes
    3. Toutes les données dans un tableau markdown formaté
    4. Totaux, moyennes ou calculs visibles
    5. Notes de bas de tableau"""
    elif type_image == "DIAGRAMME":
        rules = """**REGLES POUR UN DIAGRAMME*
    1. Type de diagramme (flux, organigramme, etc.)
    2. Éléments principaux et leurs relations
    3. Valeurs numériques ou métriques affichées
    4. Processus ou hiérarchie représentée
    5. Annotations importantes"""
    elif type_image == "ORGANIGRAMME":
        rules = """
    1. Structure hiérarchique (nombre de niveaux)
    2. Postes/fonctions à chaque niveau
    3. Noms des personnes si visibles
    4. Relations de subordination et liens transversaux
    5. Services/départements identifiés
    6. Effectifs mentionnés pour chaque service
    7. Reproduis la hiérarchie en markdown avec des listes imbriquées
    """
    elif type_image == "CARTE GEOGRAPHIQUE":
        rules = """
    1. Zone géographique couverte (pays, région, ville)
    2. Type de carte (politique, physique, routière, thématique)
    3. Échelle si visible
    4. Légende : couleurs, symboles, codes
    5. Données chiffrées (populations, distances, altitudes, statistiques)
    6. Villes, régions, points d'intérêt nommés
    7. Coordonnées géographiques si présentes
    8. Extrais toutes les données numériques dans un tableau
    9. Objectifs/but de la carte"""
    elif type_image == "SCHEMA TECHNIQUE":
        rules = """
    1. Type de système/mécanisme représenté
    2. Composants principaux et leur fonction
    3. Cotes, dimensions, mesures techniques
    4. Matériaux spécifiés
    5. Flux (électrique, hydraulique, mécanique)
    6. Références techniques (normes, codes)
    7. Annotations et spécifications
    8. Organise les données techniques en tableau structuré
    """
    elif type_image == "SCAN DE DOCUMENT":
        rules = """
    1. Type de document (facture, contrat, rapport, etc.)
    2. En-têtes et informations d'identification
    3. Tous les montants, dates, références numériques
    4. Données tabulaires si présentes
    5. Signatures, tampons, mentions légales
    6. Structure du document (sections, paragraphes)
    7. Privilégie la transcription fidèle du contenu textuel et numérique"""
    elif type_image == "PHOTO":
        rules = """
    1. Sujet principal et contexte
    2. Éléments textuels visibles (panneaux, étiquettes)
    3. Données chiffrées présentes (prix, dates, mesures)
    4. Informations factuelles observables
    5. Localisation ou indices géographiques si identifiables
    6. Objets, personnes, environnement
    7. Focus sur les informations extractibles plutôt que l'esthétique"""
    elif type_image == "LOGO":
        rules = """
    1. Nom de la marque/entreprise si visible
    2. Type de logo (texte, symbole, combiné)
    3. Éléments graphiques principaux (formes, symboles)
    4. Palette de couleurs utilisée
    5. Typographie (style de police si texte)
    6. Dimensions/proportions si indiquées
    7. Versions présentes (couleur, noir&blanc, variations)
    8. Usage ou contexte d'application
    9. Décris de manière objective sans interprétation subjective"""
    else:
        rules = """
    1. Décris ce que tu vois de façon détaillée.
    2. Identifie les éléments visibles (textes, chiffres, objets).
    3. Fournis un résumé structuré si possible.
    4. Reste objectif et précis dans la description.
    """
    
    if context==None:
        prompt = f"""Analyse cette image et fournis une description complète. 
        # Description général:
            - Paragraphe descriptif le contenu de l'image
            - Contexte et sujet principal
        # Description détaillé
        {rules}"""
        
    
    else:
        prompt = f"""Analyse cette image et fournis une description complète en t'appuyant sur son contexte : {context}
        # Description général:
            - Paragraphe descriptif le contenu de l'image
            - Contexte et sujet principal
        # Description détaillé
        {rules}"""
    
    return prompt



class ImageAnalyzer:
    """Image analysis via Ollama with minimal JSON output."""


    def __init__(self, config_server):
        self.agent = get_Agent(config_server, image_description=True)
        self.config_server=config_server 

    def analyze_bytes(self, image_bytes: bytes, context) -> ImageDescription:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{image_b64}"

        if self.config_server["params_host_llm"]["type"]=="ollama":
            temperature=0
        else:
            temperature=1
        print("test")
        category=self.agent.predict_image(prompt=classification_prompt,
                                          data_url=data_url,
                                          json_format=ImageClassification,
                                          temperature=temperature)
        print(category)
        description_prompt=get_prompt(category.category, context)
        response=self.agent.predict_image(prompt=description_prompt,
                                          data_url=data_url,
                                          json_format=ImageDescription,
                                          temperature=temperature)


        return response

def json_to_markdown(data: ImageDescription, index: int) -> list[str]:
    lines = []
    description = data.description

    lines.append(f"**Description** : {description}\n\n")
    lines.append(f"<IMAGE {index} -->")

    return lines






class DoclingConverter:
    
    def __init__(self, config_server):
        self.converter = self._init_converter()
        self.analyzer = ImageAnalyzer(config_server)

    def _init_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions(
            generate_picture_images=True,
            generate_page_images=False,
            keep_original_image_size=False,
            export_images=False,
            image_dpi=1200,
            images_scale=5,
            accelerator_options=AcceleratorOptions(
                                num_threads=4, 
                                device=AcceleratorDevice.AUTO
                            )
        )
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def convert(self, input_file: str, image_description: bool = True) -> str | None:
        """Converts a PDF to enriched Markdown and returns the content as a string."""

        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"File '{input_path}' doesn't exist")
            return None

        try:
            result = self.converter.convert(str(input_path))
            markdown_text = result.document.export_to_markdown()

            if image_description:
                final_markdown = self.process_markdown_with_images(markdown_text, result)
            else:
                final_markdown = markdown_text

            logger.info(f"Markdown conversion success for: {input_path.name}")
            return final_markdown

        except ConversionError as e:
            logger.error(f"Unsupported format for '{input_path}': {e}")
        except Exception as e:
            logger.error(f"Erreur with '{input_path}': {e}")

        return None

    def process_markdown_with_images(self, markdown_text: str, result) -> str:
        """Replaces <!-- image --> tags in the Markdown with image descriptions."""

        lines = markdown_text.splitlines(keepends=True)
        final_lines = []

        
        
        pictures_and_stacks = [
    (item, tuple(stack))  # or stack.copy()
    for item, stack in result.document._iterate_items_with_stack(traverse_pictures=True)
    if isinstance(item, PictureItem)
]

        img_index = 0
        line_index = 0

        while line_index < len(lines):
            line = lines[line_index]

            if "<!-- image -->" in line and img_index < len(pictures_and_stacks):
                picture_item, stack = pictures_and_stacks[img_index]
                img_index += 1

                
                image = picture_item.get_image(result)
                print(np.array(image).shape)
                try:
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")

                    
                    header_node = None
                    if stack:  
                        try:
                            header_node = nearest_header_from_stack(result.document, stack)
                        except Exception as e:
                            logger.warning(f"Unable to find the header for image '{img_index}' : {e}")

                    title = getattr(header_node, "text", None) if header_node else None
                    
                    description = self.analyzer.analyze_bytes(buffer.getvalue(), context=title)

                    final_lines.append(f"<!-- IMAGE {img_index} -->\n")
                    final_lines.extend(json_to_markdown(description, img_index))

                except Exception as e:
                    logger.error(f"Error image {img_index}: {e}")
                    final_lines.append(line)

            else:
                final_lines.append(line)

            line_index += 1

        return "".join(final_lines)




# The following functions are used to locate the header of the section where an image is found, 
# so it can be sent as context to the VLM that describes it.


def is_header(node) -> bool:
    # Headers are TextItem with a SECTION_HEADER or TITLE label
    return isinstance(node, TextItem) and node.label in {
        DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE
    }



def rightmost_header_in_subtree(doc, node):
    """If a child is a GroupItem, go down to the bottom-right to find
    the most recent header in this subtree."""

    if not isinstance(node, GroupItem):
        return None
    cur = node
    while isinstance(cur, GroupItem) and cur.children:
        # go to the rightmost child

        cur = cur.children[-1].resolve(doc)
        if is_header(cur):
            return cur
    return None


def nearest_header_from_stack(doc, stack):
    """Searches for the closest section header upstream of the item addressed by `stack`."""
# starting point: parent and local index

    parent = doc.body._get_parent_ref(doc, stack).resolve(doc)
    idx = stack[-1]

    # 1) Search in the previous siblings

    for j in range(idx - 1, -1, -1):
        node = parent.children[j].resolve(doc)
        if is_header(node):
            return node
        # if it's a group, check if it contains a header "deep on the right"

        h = rightmost_header_in_subtree(doc, node)
        if h:
            return h

    # 2) Nothing found? Go up and search among the parent's previous siblings

    cur_parent_ref = getattr(parent, "parent", None)
    cur_stack = stack[:-1]  # remove the last index

    while cur_parent_ref:
        cur_parent = cur_parent_ref.resolve(doc)
        if not hasattr(cur_parent, "children") or len(cur_stack) == 0:
            break
        parent_index_in_grandparent = cur_stack[-1]
        for j in range(parent_index_in_grandparent - 1, -1, -1):
            node = cur_parent.children[j].resolve(doc)
            if is_header(node):
                return node
            h = rightmost_header_in_subtree(doc, node)
            if h:
                return h
        # go up again

        cur_stack = cur_stack[:-1]
        cur_parent_ref = getattr(cur_parent, "parent", None)

    return None  # no header found 


