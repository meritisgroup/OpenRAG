import streamlit as st
import os
import shutil
import pandas as pd
from elasticsearch import Elasticsearch
import re
from backend.utils.open_doc import Opener

st.markdown("# Database Manager")

st.markdown("## Database creation")
left, right = st.columns([6, 1], vertical_alignment="bottom")
st.session_state["databases"]["new_name"] = left.text_input(
    label="**Database name**",
    placeholder="Database name: alphanumeric characters, underscores and hyphens only",
    label_visibility="collapsed",
)


if right.button(label="Create DataBase", type="primary", use_container_width=True):
    if st.session_state["databases"]["new_name"] in os.listdir("./data/databases/"):
        st.error(
            f"Database named {st.session_state['databases']['new_name']} already exists",
            icon="🚨",
        )
    elif not bool(
        re.fullmatch(r"^[a-z0-9_-]+$", st.session_state["databases"]["new_name"])
    ):
        st.error(
            "Invalid RAG name, only alphanumeric characters, underscores, lowercase letters and hyphens are allowed",
            icon="🚨",
        )
    else:
        os.mkdir(f"./data/databases/{st.session_state['databases']['new_name']}")
        st.session_state["all_databases"].append(
            st.session_state["databases"]["new_name"]
        )
        st.rerun()


st.markdown("## Database visualization")

left, right = st.columns([6, 1], vertical_alignment="bottom")
database_name = left.selectbox(
    label="Choose database",
    options=st.session_state["all_databases"],
    label_visibility="collapsed",
)
if right.button(label="Delete DataBase", type="primary", use_container_width=True):

    shutil.rmtree(f"./data/databases/{database_name}")
    st.session_state["all_databases"].remove(database_name)

    es = Elasticsearch([st.session_state["config_server"]["params_vectorbase"]["url"]],
                    basic_auth=(st.session_state["config_server"]["params_vectorbase"]["auth"][0],
                                st.session_state["config_server"]["params_vectorbase"]["auth"][1]))
    for index_name in es.indices.get_alias(index="*"):
        if database_name in index_name:
            es.indices.delete(index=index_name)
            print(f"{index_name} successfully deleted")
    
    for database in os.listdir("./storage"):
        if database_name in database:
            os.remove("./storage/" + database)
            print(f"{database} successfully deleted")

    if st.session_state["chat_database_name"] == database_name:
        st.session_state["chat_database_name"] = None
    if st.session_state["benchmark_database"] == database_name:
        st.session_state["benchmark_database"] = None
    database_name = None
    st.rerun()

if "new_doc" not in st.session_state:
    st.session_state["new_doc"] = False

if database_name is not None:
    database_path = f"./data/databases/{database_name}"
    document_list = os.listdir(f"./data/databases/{database_name}")
    display_db = pd.DataFrame(data={"Doc Name": document_list})
    st.write(display_db)

    if "new_doc" not in st.session_state:
        st.session_state["new_doc"] = False

    def new_doc():
        st.session_state["new_doc"] = True


if "document_deleted" not in st.session_state:
    st.session_state["document_deleted"] = False

if 'document_list' in globals() and document_list:
    file_to_delete = st.selectbox("Select a document to delete:", document_list)
    if st.button("Delete selected document"):
        os.remove(os.path.join(database_path, file_to_delete))
        st.success(f"Deleted {file_to_delete}")
        st.session_state["document_deleted"] = True
        st.rerun()
else:
    st.info("No documents to display or delete.")

if st.session_state["document_deleted"]:
    st.success("A document has been deleted. Don’t forget to reset the indexing!")
    st.session_state["document_deleted"] = False


if 'new_doc' in globals() and callable(globals()['new_doc']):
    uploaded_files = st.file_uploader(
        "**Add document to DataBase**",
        type=["pdf", "docx", "xlsx", "txt", "pptx"],
        accept_multiple_files=True,
        label_visibility="visible",
        on_change=new_doc,
    )
    

if st.session_state["new_doc"]:
    database_path = os.path.join("./data/databases", database_name)
    for file in uploaded_files:
        file_path = os.path.join(database_path, file.name)
        if file.name not in os.listdir(database_path):
            with open(file_path, "wb") as f:
                f.write(file.read())
        doc = Opener(save=False).open_doc(path_file=file_path).strip()
        if len(doc) == 0:
            os.remove(file_path)
            st.warning(
                f"Doc {file.name} was not added to database because it is empty.  \n  Please note that images are not read, even if they are in a supported filetype"
            )
            st.session_state["new_doc"] = False
        else:
            st.session_state["new_doc"] = False
    st.session_state["new_doc"] = False
    st.rerun()
