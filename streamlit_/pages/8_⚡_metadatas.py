import streamlit as st
import json
import os
import pandas as pd

from streamlit_.utils.metadatas_funcs import (
    save_new_metadata_key,
    load_metadatas_keys,
    save_individual_modification,
    save_global_modification,
    get_documents_of_database,
    get_metadatas,
    delete_metadata,
    delete_button,
)

st.markdown("# Metadatas")

st.write("Select your database")

left, right = st.columns([6, 1], vertical_alignment="bottom")


def change_metadatas_keys():
    db_name = st.session_state.db_name
    st.session_state.metadata = load_metadatas_keys(db_name)


database_name = left.selectbox(
    label="Choose database",
    options=st.session_state["all_databases"],
    label_visibility="collapsed",
    key="db_name",
    on_change=change_metadatas_keys,
)

if "metadata" not in st.session_state:
    st.session_state.metadata = load_metadatas_keys(database_name)

st.markdown("## Add new metadata key")

left, right = st.columns([6, 1], vertical_alignment="bottom")

st.session_state["new_metadata"] = left.text_input(
    label="**New metadata**",
    placeholder="Metadata key : alphanumeric characters, underscores and hyphens only",
    label_visibility="collapsed",
)

if right.button(label="Add metadata key", type="primary", use_container_width=True):
    st.session_state["metadata"].append(st.session_state["new_metadata"])
    save_new_metadata_key(
        database_name=database_name, new_key=st.session_state["new_metadata"]
    )
    st.session_state["metadata"] = load_metadatas_keys(database_name)

if "global_ui" not in st.session_state:
    st.session_state.global_ui = False
if "individual_ui" not in st.session_state:
    st.session_state.individual_ui = False
if "delete_ui" not in st.session_state:
    st.session_state.delete_ui = False
if "button" not in st.session_state:
    st.session_state.button = True

if st.session_state["metadata"] != []:
    st.markdown("## Modify metadatas")

    left, middle, right_col = st.columns([4, 2, 2], vertical_alignment="bottom")
    keys_available = load_metadatas_keys(database_name)
    key_to_change = left.selectbox(
        label="Choose the metadata to change", options=keys_available
    )

    if "individual_mod_active" not in st.session_state:
        st.session_state.individual_mod_active = False

    if "global_mod_active" not in st.session_state:
        st.session_state.individual_mod_active = False

    if middle.button(
        label="Global modification", type="primary", use_container_width=True
    ):

        st.session_state.global_ui = True
        st.session_state.global_mod_active = True
        st.session_state.individual_mod_active = False
        st.session_state.individual_ui = False
        st.rerun()

    if right_col.button(
        label="Individual modification", type="primary", use_container_width=True
    ):
        st.session_state.global_mod_active = False
        st.session_state.global_ui = False
        st.session_state.individual_mod_active = True
        st.session_state.individual_ui = True
        st.rerun()

if st.session_state.global_ui:
    new_input = st.text_input(
        label="*The metadata to add",
        placeholder=f"**{key_to_change}**",
        label_visibility="collapsed",
    )
    document_list = get_documents_of_database(database_name)
    col1, col2, col3 = st.columns([3, 2, 1])
    selected_docs = {}
    with col1:
        st.markdown("**Document**")
    with col2:
        st.markdown("**Already registered**")
    with col3:
        st.markdown("**Select to change**")

    metadatas_already_in_place = get_metadatas(database_name)
    for doc in document_list:
        with col1:
            st.write(f"{doc}")
        with col2:
            metadatas_already_in_place["documents"][doc][key_to_change]
        with col3:
            selected_docs[doc] = st.checkbox(
                "",
                key=f"select_{doc}",
                value=False,
            )

    left, right = col1, col2 = st.columns([6, 1], vertical_alignment="bottom")
    if right.button(
        label="Save",
        type="primary",
        use_container_width=True,
    ):
        save_global_modification(
            database_name,
            key_to_change,
            new_input,
            selected_docs,
        )
        st.session_state.global_mod_active = False
        for k in list(st.session_state.keys()):
            if k.startswith("select_"):
                del st.session_state[k]
        st.session_state.global_ui = False
        st.rerun()


if st.session_state.individual_ui:
    if st.session_state.individual_mod_active and database_name is not None:
        document_list = get_documents_of_database(database_name)
        display_db = pd.DataFrame(data={"Doc Name": document_list})

        metadata_inputs = {}
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            st.markdown("**Document**")
        with col2:
            st.markdown("**Already registered**")
        with col3:
            st.markdown("**New metadata**")

        for doc in document_list:
            metadatas_already_in_place = get_metadatas(database_name)
            with col1:
                st.markdown(
                    f"<p style='margin: 0; padding-top: 3px;'>{doc}</p>",
                    unsafe_allow_html=True,
                )
            with col2:
                metadatas_already_in_place["documents"][doc][key_to_change]
            with col3:
                metadata_inputs[doc] = st.text_input(
                    label="",
                    placeholder=f"**{key_to_change}**",
                    key=f"input_{doc}",
                    label_visibility="collapsed",
                )
        left, right = col1, col2 = st.columns([6, 1], vertical_alignment="bottom")
        if right.button(
            label="Save",
            type="primary",
            use_container_width=True,
            on_click=save_individual_modification(
                database_name, key_to_change, metadata_inputs
            ),
        ):
            for k in list(st.session_state.keys()):
                if k.startswith("input_"):
                    del st.session_state[k]
            st.session_state.individual_ui = False
            st.session_state.individual_mod_active = False
            st.rerun()

if st.session_state["metadata"] != []:

    st.markdown("## Delete a metadata")
    if st.session_state.button:
        if st.button("Click to display the UI", type="secondary"):
            delete_button()
            st.session_state.delete_ui = True
            st.rerun()

    if st.session_state.delete_ui:
        left, middle, right = st.columns([4, 1, 1], vertical_alignment="bottom")
        keys_available = load_metadatas_keys(database_name)
        key_to_delete = left.selectbox(
            label="Choose the metadata to delete", options=keys_available
        )

        if middle.button(
            label="Delete",
            type="primary",
            use_container_width=True,
        ):
            delete_metadata(database_name, key_to_delete)
            st.session_state.delete_ui = False
            st.session_state.button = True
            st.rerun()

        if right.button(
            label="Cancel",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.delete_ui = False
            st.session_state.button = True
            st.rerun()
