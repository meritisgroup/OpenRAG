import streamlit as st
import json
import pandas as pd
import os


def get_database_path(database_name):
    return f"./data/databases/{database_name}/metadatas.json"


def get_documents(metadatas: pd.DataFrame):
    return metadatas["documents"].keys()


def get_documents_of_database(database_name):
    path = get_database_path(database_name)
    with open(path, "r") as f:
        metadatas: pd.DataFrame = json.load(f)
    return list(metadatas.get("documents", {}).keys())


def save_new_metadata_key(database_name, new_key):
    database_metadata_path = get_database_path(database_name)

    with open(database_metadata_path, "r") as f:
        metadatas = json.load(f)

    if "keys" not in metadatas:
        metadatas["keys"] = []
    if new_key not in metadatas["keys"]:
        metadatas["keys"].append(new_key)

    for doc in metadatas["documents"]:
        metadatas["documents"][doc][new_key] = None

    with open(database_metadata_path, "w") as f:
        json.dump(metadatas, f, indent=4)


def get_metadatas(database_name):
    path = get_database_path(database_name)
    with open(path, "r") as f:
        metadatas = json.load(f)
    return metadatas


def load_metadatas_keys(database_name):

    metadata_path = get_database_path(database_name)
    with open(metadata_path, "r") as f:
        metadatas = json.load(f)
    return metadatas["keys"]


def save_individual_modification(database_name, key, metadatas_inputs):

    database_path = get_database_path(database_name)
    with open(database_path, "r") as f:
        metadatas = json.load(f)
        documents = get_documents(metadatas)

        for doc in documents:
            if metadatas_inputs[doc] != "":
                metadatas["documents"][doc][key] = metadatas_inputs[doc]

    with open(database_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, indent=4)


def save_global_modification(database_name, key, new_input, metadata_input):
    database_path = get_database_path(database_name)
    with open(database_path, "r") as f:
        metadatas = json.load(f)
        documents = get_documents(metadatas)

        for doc in documents:
            if metadata_input[doc]:
                metadatas["documents"][doc][key] = new_input

        with open(database_path, "w", encoding="utf-8") as f:
            json.dump(metadatas, f, indent=4)


def delete_metadata(database_name, key):
    database_path = get_database_path(database_name)
    with open(database_path, "r") as f:
        metadatas: pd.DataFrame = json.load(f)

    if key in metadatas.get("keys", []):
        metadatas["keys"].remove(key)

    for doc, fields in metadatas["documents"].items():
        fields.pop(key)

    with open(database_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, indent=4)


def add_documents_metadata(database_name):
    database_path = f"./data/databases/{database_name}"
    metadata_path = get_database_path(database_name)

    all_files = os.listdir(database_path)
    all_docs = [f for f in all_files if f != "metadatas.json"]

    with open(metadata_path, "r") as f:
        metadatas = json.load(f)

    keys = load_metadatas_keys(database_name)

    saved_docs = list(metadatas["documents"].keys())
    for saved_doc in saved_docs:
        if saved_docs not in all_docs:
            metadatas["documents"].pop(saved_doc)

    for doc in all_docs:

        if doc not in metadatas["documents"]:
            metadatas["documents"][doc] = {}

        if keys:
            for key in keys:
                if key not in metadatas["documents"][doc]:
                    metadatas["documents"][doc][key] = None

    with open(metadata_path, "w") as f:
        json.dump(metadatas, f, indent=4)


def init_metadata(database_name):
    metadata_path = get_database_path(database_name)
    if not os.path.exists(metadata_path):
        metadatas = {"keys": [], "documents": {}}
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadatas, f, indent=4, ensure_ascii=False)


def delete_database_metadata(database_name):
    pass


def delete_button():
    st.session_state.button = not st.session_state.button
