import streamlit as st
from streamlit_.utils.benchmark_funcs import (
    get_report_path,
    run_indexation_benchmark,
    generate_benchmark,
    generate_only_answers,
    generate_only_contexts,
)
import subprocess
import json
import sys
import os
import argparse
import pickle
import platform


def task(
    reset_index,
    reset_preprocess,
    report_dir,
    type_bench,
    config_server,
    queries_doc_name,
    databases,
    session_state=None,
    background=False,
):

    rag_agents, rag_names = run_indexation_benchmark(
        reset_index=reset_index,
        reset_preprocess=reset_preprocess,
        databases=databases,
        report_dir=report_dir,
        session_state=session_state,
    )
    with st.spinner(
        f"**Generating benchmark for the following RAGs:** {rag_names}",
        show_time=True,
    ):
        if len(rag_agents) > 1:
            if type_bench == "answers":
                generate_only_answers(rag_names, rag_agents, report_dir=report_dir)
            elif type_bench == "contexts":
                generate_only_contexts(
                    rag_names=rag_names, rag_agents=rag_agents, report_dir=report_dir
                )
            elif type_bench == "full_bench":
                generate_benchmark(
                    rag_names,
                    rag_agents,
                    databases=databases,
                    config_server=config_server,
                    report_dir=report_dir,
                    queries_doc_mane=queries_doc_name,
                    session_state=session_state,
                )
                if not background:
                    databases = st.session_state["benchmark_database"]
                    markdown_text = "Benchmark runned on the following database:\n"
                    for db in databases:
                        markdown_text += f"- {db}\n"
                    st.markdown(markdown_text)

        if not background:
            st.session_state["benchmark_clicked"] = False
            st.set_option("client.showSidebarNavigation", True)
            st.rerun()


def run_benchmark(
    type_bench, reset_index=False, reset_preprocess=False, background=False
):
    rag_to_run = [
        rag
        for rag in st.session_state["benchmark"]["rags"].keys()
        if st.session_state["benchmark"]["rags"][rag]
    ]
    if len(rag_to_run) > 1 or type_bench != "full_bench":
        databases = st.session_state["benchmark_database"]
        report_dir = get_report_path()
        config_server = st.session_state["config_server"]
        queries_doc_name = st.session_state["benchmark"]["queries_doc_name"]

        if background:
            log_file = "test.log"
            log_f = open(log_file, "w")
            system = platform.system()

            config_server_str = json.dumps(config_server)
            databases_str = json.dumps(databases)
            session_state = dict(st.session_state)

            script_path = os.path.abspath(__file__)
            session_state_file = os.path.join(
                os.path.dirname(script_path), "session_state_background.pkl"
            )
            with open(session_state_file, "wb") as f:
                pickle.dump(session_state, f)

            python_exe = sys.executable

            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(script_path))
            )
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

            args = [
                python_exe,
                script_path,
                "--reset_index",
                str(reset_index),
                "--reset_preprocess",
                str(reset_preprocess),
                "--report_dir",
                report_dir,
                "--type_bench",
                type_bench,
                "--config_server",
                config_server_str,
                "--queries_doc_name",
                queries_doc_name,
                "--databases",
                databases_str,
                "--session_state_file",
                session_state_file,
            ]
            if system in ("Linux", "Darwin"):
                process = subprocess.Popen(
                    args,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    preexec_fn=os.setpgrp,
                    close_fds=True,
                )

            elif system == "Windows":
                DETACHED_PROCESS = 0x00000008
                process = subprocess.Popen(
                    args,
                    env=env,
                    stdout=log_f,
                    stderr=log_f,
                    stdin=subprocess.DEVNULL,
                    close_fds=True,
                    creationflags=DETACHED_PROCESS,
                )
        else:
            task(
                reset_index=reset_index,
                reset_preprocess=reset_preprocess,
                report_dir=report_dir,
                type_bench=type_bench,
                config_server=config_server,
                queries_doc_name=queries_doc_name,
                databases=databases,
            )

        st.session_state["benchmark_clicked"] = False
        st.set_option("client.showSidebarNavigation", True)
        st.rerun()
    else:
        st.session_state["benchmark_clicked"] = False
        st.error("Choose at least 2 rags methods")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run benchmark task in background.")
    parser.add_argument("--reset_index", type=str, default="False")
    parser.add_argument("--reset_preprocess", type=str, default="False")
    parser.add_argument("--report_dir", type=str, required=True)
    parser.add_argument("--type_bench", type=str, required=True)
    parser.add_argument("--config_server", type=str, required=True)
    parser.add_argument("--session_state_file", type=str, required=True)
    parser.add_argument("--queries_doc_name", type=str, required=True)
    parser.add_argument(
        "--databases", type=str, required=True, help="Comma-separated list of databases"
    )

    args = parser.parse_args()

    config_server = json.loads(args.config_server)
    databases = json.loads(args.databases)
    reset_index = args.reset_index
    report_dir = args.report_dir
    type_bench = args.type_bench

    with open(args.session_state_file, "rb") as f:
        session_state = pickle.load(f)

    task(
        reset_index=reset_index,
        report_dir=args.report_dir,
        type_bench=args.type_bench,
        config_server=config_server,
        queries_doc_name=args.queries_doc_name,
        databases=databases,
        session_state=session_state,
        background=True,
    )
