import streamlit as st
import json

st.markdown("# Advanced Configurations")

default = "default"
if "all_system_prompt" not in st.session_state:
    st.session_state["all_system_prompt"] = st.session_state["config_server"][
        "all_system_prompt"
    ]

if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = {}
    st.session_state["system_prompt"]["new_name"] = ""
    st.session_state["system_prompt"]["new_prompt"] = ""

if "prompt_added" not in st.session_state:
    st.session_state["prompt_added"] = False

if "adding_prompt" not in st.session_state:
    st.session_state["adding_prompt"] = False

if "modifying_prompt" not in st.session_state:
    st.session_state["modifying_prompt"] = False

if "editing_prompt" not in st.session_state:
    st.session_state["editing_prompt"] = None


def save_modification():
    st.session_state["config_server"]["all_system_prompt"] = st.session_state[
        "all_system_prompt"
    ]
    st.session_state["config_server"]["chunk_length"] = st.session_state["chunk_length"]
    with open("streamlit_/utils/base_config_server.json", "w") as file:
        json.dump(st.session_state["config_server"], file, indent=4)


st.markdown("## System prompt creation")

add_new_prompt = st.button(label="Add a new prompt system")
if add_new_prompt:
    st.session_state["adding_prompt"] = True

if st.session_state["adding_prompt"]:
    left, right = st.columns([6, 1], vertical_alignment="bottom")
    name = left.text_input(
        label="**System Prompt name**",
        placeholder="System prompt name: alphanumeric characters, underscores and hyphens only",
        label_visibility="collapsed",
    )

    prompt = left.text_area(
        label="**Your system prompt**",
        placeholder="Your system prompt",
        label_visibility="collapsed",
    )

    add_prompt_btn = right.button(
        label="Add prompt", type="primary", use_container_width=True
    )

    if add_prompt_btn:
        if name in st.session_state["all_system_prompt"].keys() or '"' in name:
            st.error(
                "This name is already taken or incorrect, please choose a new name"
            )
        else:
            if name and prompt:
                st.session_state["all_system_prompt"][name] = prompt
                save_modification()
                st.success("Prompt added!")
                st.session_state["system_prompt_name"] = ""
                st.session_state["system_prompt_prompt"] = ""
                st.session_state["adding_prompt"] = False
            else:
                st.error("Please fill in both the name and the prompt before adding.")


st.markdown("## Modify your prompts")
modify_prompt = st.button(label="Modify a prompt system")
if modify_prompt:
    st.session_state["modifying_prompt"] = True

if st.session_state["modifying_prompt"]:
    left, right = st.columns([6, 1], vertical_alignment="bottom")
    system_prompt_name = left.selectbox(
        label="Your prompts", options=st.session_state["all_system_prompt"].keys()
    )

    modify_prompt_btn = right.button(
        label="Modify prompt",
        type="primary",
        use_container_width=True,
    )

    if modify_prompt_btn:
        st.session_state["editing_prompt"] = system_prompt_name

    if st.session_state["editing_prompt"]:
        left, right = st.columns([6, 1], vertical_alignment="bottom")
        prompt_to_edit = st.session_state["editing_prompt"]
        new_prompt = left.text_area(
            label="system prompt",
            value=st.session_state["all_system_prompt"][prompt_to_edit],
        )
        changes = right.button(
            label="Save changes",
            type="primary",
            use_container_width=True,
            disabled=system_prompt_name == default,
        )

        if changes:
            st.session_state["all_system_prompt"][prompt_to_edit] = new_prompt
            st.success("Prompt changed!")

            st.session_state["editing_prompt"] = None
            st.session_state["modifying_prompt"] = False
            st.rerun()

keys_without_default = [
    k for k in st.session_state["all_system_prompt"].keys() if k != default
]

if keys_without_default:
    st.markdown("## Delete a prompt")
    prompt_to_delete = st.selectbox(
        "Select a document to delete:", keys_without_default
    )
    if st.button("Delete selected prompt"):
        st.session_state["editing_prompt"] = default
        st.session_state["modifying_prompt"] = False
        st.session_state["adding_prompt"] = False
        st.session_state["all_system_prompt"].pop(prompt_to_delete, None)
        save_modification()
        st.success(f"Prompt deleted")
        st.rerun()

st.markdown("## Chunk length")
if "chunk_length" not in st.session_state:
    st.session_state["chunk_length"] = st.session_state["config_server"]["chunk_length"]

if "numeric" not in st.session_state:
    st.session_state["numeric"] = st.session_state["config_server"]["chunk_length"]

if "indexing" not in st.session_state:
    st.session_state["indexing"] = False


def update_slider_from_num():
    st.session_state["chunk_length"] = st.session_state["numeric"]
    st.session_state["indexing"] = True
    save_modification()


def update_num_from_slider():
    st.session_state["numeric"] = st.session_state["chunk_length"]
    st.session_state["indexing"] = True
    save_modification()


st.number_input(
    "Chunk length",
    value=st.session_state["numeric"],
    key="numeric",
    on_change=update_slider_from_num,
    step=10,
)

st.slider(
    label="**Choose length of chunks for indexing phases:**",
    min_value=0,
    max_value=2000,
    step=10,
    value=st.session_state["chunk_length"],
    help=""" Keep in mind that too long or too short chunks can make the retrieval harder and decrease accuracy """,
    key="chunk_length",
    on_change=update_num_from_slider,
)

if st.session_state["indexing"]:
    st.warning("You changed the chunk length, don't forget to rerun the indexing 😉")
