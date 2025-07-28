import streamlit as st
import os
import shutil
import pandas as pd
from elasticsearch import Elasticsearch
import re
from backend.utils.open_doc import Opener
import zipfile
import io

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

    es = Elasticsearch(
        [st.session_state["config_server"]["params_vectorbase"]["url"]],
        basic_auth=(
            st.session_state["config_server"]["params_vectorbase"]["auth"][0],
            st.session_state["config_server"]["params_vectorbase"]["auth"][1],
        ),
    )
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


if "processing_warnings" not in st.session_state:
    st.session_state["processing_warnings"] = []

if st.session_state["processing_warnings"]:
    for msg in st.session_state["processing_warnings"]:
        st.warning(msg)
    # Let the user clear them when they're done
    if st.button("Dismiss all warnings"):
        st.session_state["processing_warnings"] = []
        st.rerun()

if "document_deleted" not in st.session_state:
    st.session_state["document_deleted"] = False

if "document_list" in globals() and document_list:
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

SUPPORTED = ["pdf", "docx", "xlsx", "txt", "pptx", "zip"]

if "new_doc" in globals() and callable(globals()["new_doc"]):
    uploaded_files = st.file_uploader(
        "**Add document to DataBase**",
        type=SUPPORTED,
        accept_multiple_files=True,
        label_visibility="visible",
        on_change=new_doc,
    )


def process_file(file_name: str, file_data: bytes):
    file_path = os.path.join(database_path, file_name)
    if file_name not in os.listdir(database_path):
        with open(file_path, "wb") as f:
            f.write(file_data)
    doc = Opener(save=False).open_doc(path_file=file_path).strip()
    if not doc:
        os.remove(file_path)
        st.warning(
            f"Doc {file_name} was not added to database because it is empty.  \n  Please note that images are not read, even if they are in a supported filetype"
        )


if st.session_state["new_doc"]:
    database_path = os.path.join("./data/databases", database_name)
    for file in uploaded_files:
        file_name: str = file.name
        file_data: bytes = file.read()
        file_path = os.path.join(database_path, file.name)

        if file_name.lower().endswith(".zip"):
            unsupported = []
            try:
                with zipfile.ZipFile(io.BytesIO(file_data)) as zip_folder:
                    for info in zip_folder.infolist():
                        if info.is_dir():
                            continue

                        member = info.filename
                        ext = member.rsplit(".", 1)[-1].lower()
                        if ext in SUPPORTED:
                            data = zip_folder.read(member)
                            safe_name = os.path.basename(member)
                            process_file(safe_name, data)
                        else:
                            unsupported.append(member)

                if unsupported:
                    st.session_state["processing_warnings"].append(
                        "The following items in the ZIP were skipped because they "
                        "are either folders or unsupported file types:\n\n"
                        + "\n".join(f"• {u}" for u in unsupported)
                        + "\n\nPlease include only PDF, DOCX, XLSX, TXT, or PPTX files."
                    )

            except Exception as e:
                st.session_state["processing_warnings"].append(
                    "The uploaded ZIP is corrupted or not a valid archive. "
                    "Please upload a proper .zip containing only supported files."
                    f"{e}"
                )
        else:
            ext = file_name.rsplit(".", 1)[-1].lower()
            if ext in SUPPORTED:
                try:
                    process_file(file_name, file_data)
                except Exception as e:
                    st.session_state["processing_warnings"].append(
                        f"We encountered a problem with the file {file_name}"
                    )
            else:
                st.session_state["processing_warnings"].append(
                    f"File `{file_name}` is not a supported type."
                )

    st.session_state["new_doc"] = False
    st.rerun()
