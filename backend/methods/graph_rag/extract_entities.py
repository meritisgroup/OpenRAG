
from __future__ import annotations
import os
import json
import pandas as pd
from typing import List


from ...utils.agent import Agent
from ...utils.open_doc import Opener, MarkdownOpener
from ...utils.splitter import TextSplitter
from ...utils.base_classes import Splitter
from ...utils.splitter import MarkdownHeaderTextSplitter
from ...database.rag_classes import Document, Chunk, Entity, Relation
from .prompts import PROMPTS
from ...utils.progress import ProgressBar
import re
import html
import numpy as np





### ----- EXTRACT ENTITIES FROM A TEXT ----- ###


def extract_entities_relations(
    agent: Agent,
    chunks: list[Chunk],
    doc_name: str,
    language: str = "EN",
) -> tuple[list[Entity], list[Relation]]:
    """
    Using a LLM agent to extract entities and relations from the chunk
    """
    output_language = "english"
    if language == "FR":
        output_language == "french"

    context_base = dict(
        tuple_delimiter=PROMPTS[language]["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS[language]["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS[language]["DEFAULT_COMPLETION_DELIMITER"],
        language=output_language,
    )
 
    prompts, system_prompts = _get_extract_prompts(chunks, context_base,language=language)

    tokens = 0
    taille_batch = 50
    outputs = None
    data_range = range(0, len(prompts), taille_batch)
    progress_bar = ProgressBar(total=len(data_range))
    k = 0
    for i in data_range:
        results = agent.multiple_predict(prompts=prompts[i:i + taille_batch],
                                         system_prompts=system_prompts[i:i + taille_batch])
        k+=1
        progress_bar.update(k-1, text="Extraction done {}".format(np.round((k/len(data_range))*100,2)))
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

    progress_bar.update(k, text="Extraction done {}".format(np.round((k/len(data_range))*100,2)))
    
    llm_outputs = outputs["texts"]

    input_tokens = outputs["nb_input_tokens"]
    output_tokens = outputs["nb_output_tokens"]

    if isinstance(llm_outputs[0], tuple):
        llm_outputs = [llm_outputs]

    entities, relations = _clean_extraction_outputs(
        llm_outputs, chunks, context_base, doc_name
    )

    return entities, relations, input_tokens, output_tokens


def _get_extract_prompts(
    chunks: list[Chunk], context_base, language
) -> tuple[list[str], list[str]]:
    prompt_template = PROMPTS[language]["extraction_text"]["QUERY_TEMPLATE"]
    system_prompt_template = PROMPTS[language]["extraction_text"]["SYSTEM_PROMPT"]

    prompts = [
        prompt_template.format(**context_base, input_text=chunk) for chunk in chunks
    ]
    system_prompts = [system_prompt_template] * len(chunks)

    return prompts, system_prompts


def _clean_extraction_outputs(
    outputs: list[str], chunks: list[Chunk], context_base, doc_name
) -> tuple[list[Entity], list[Relation]]:
    already_processed = 0
    already_entities = 0
    already_relations = 0
    new_nodes = []
    new_edges = []

    for output, chunk in zip(outputs, chunks):
        chunk_key = chunk.id

        records = _split_string_by_multi_markers(
            output,
            [
                context_base["record_delimiter"],
                context_base["completion_delimiter"],
                "\n",
            ],
        )

        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)

            record_attributes = _split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )

            try:

                if_entities = _handle_single_entity_extraction(
                    record_attributes, chunk_key, doc_name
                )

                if if_entities is not None:
                    new_nodes.append(if_entities)

                if_relation = _handle_single_relationship_extraction(
                    record_attributes, chunk_key, doc_name
                )

                if if_relation is not None:
                    new_edges.append(if_relation)
            except Exception as e:
                print("Problem to handle the record", e)

        already_processed += 1
        already_entities += len(new_nodes)
        already_relations += len(new_edges)

    return new_nodes, new_edges


def _handle_single_entity_extraction(
    record_attributes: list[str], chunk_key: int, doc_name: str
):

    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None

    entity_name = _clean_str(record_attributes[1].upper())

    if not entity_name.strip():
        return None

    entity_type = _clean_str(record_attributes[2].upper())
    entity_description = _clean_str(record_attributes[3])
    entity_source_id = chunk_key

    entity = Entity(
        name=entity_name,
        kind=entity_type,
        description=entity_description,
        chunk_id=entity_source_id,
        doc_name=doc_name,
    )

    return entity


def _handle_single_relationship_extraction(
    record_attributes: list[str], chunk_key: str, doc_name: str
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None

    source = _clean_str(record_attributes[1].upper())
    target = _clean_str(record_attributes[2].upper())
    edge_description = _clean_str(record_attributes[3])

    edge_keywords = _clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if _is_float_regex(record_attributes[-1]) else 1.0
    )
    return Relation(
        source=source,
        target=target,
        description=edge_description,
        weight=weight,
        keywords=edge_keywords,
        chunk_id=edge_source_id,
        doc_name=doc_name,
    )


def _is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def _clean_str(input) -> str:

    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())

    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def _split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:

    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]



"""
#To save chunks of a single file as txt
def save_chunks_to_file(chunks: list[Chunk], output_path: str = "chunks_output.txt"):
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(f"--- Chunk {chunk.id} ---\n")
                f.write(f"Document: {chunk.document}\n")
                f.write("Content:\n")
                f.write(chunk.text.strip() + "\n\n")
"""

