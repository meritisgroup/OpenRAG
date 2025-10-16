import streamlit as st
import pickle
import plotly.express as px
from plotly.subplots import make_subplots
from math import sqrt
from backend.evaluation.agent_evaluator import DataFramePreparator, AgentEvaluator
from backend.evaluation.prompts import PROMPTS
import pandas as pd
import os
from datetime import datetime
import json
import numpy as np
import ast

from streamlit_.utils.chat_funcs import get_chat_agent
from backend.utils.progress import ProgressBar


import base64
import random
from pydantic import BaseModel
from io import BytesIO
from backend.utils.agent import get_Agent
import fitz


color_discrete_sequence = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


question_generation_prompt = """
You are an assistant specialized in generating questions from documents.  
You are given an image of a single page from a PDF.  

Your task:  
1. Carefully read ONLY the content visible on this page.  
2. Generate ONE clear and direct question based strictly on the content of this page.  
   - The question must stand on its own: do NOT reference the page, the document, or the source (never use phrases like "according to this document" or "on this page").  
   - The question should be specific and test understanding of the page’s content.  
   - Do NOT invent information not present on this page.  
3. Provide the correct answer to the question, also written directly, without mentioning the document.  

Format your output as follows (and nothing else):  

query: <your question here>  
answer: <your answer here>
"""

class QuestionOnPage(BaseModel):
    query: str
    answer : str

class QuestionGenerator:
    """Image analysis via Ollama with minimal JSON output."""


    def __init__(self, config_server):
        self.agent = get_Agent(config_server, image_description=True)
        self.config_server=config_server 

    def generate_question(self, image_bytes: bytes) -> QuestionOnPage:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{image_b64}"

        if self.config_server["params_host_llm"]["type"]=="ollama":
            temperature=0
        else:
            temperature=1

        print("test")
    
        response=self.agent.predict_image(prompt=question_generation_prompt,
                                          data_url=data_url,
                                          json_format=QuestionOnPage,
                                          temperature=temperature)
        return response




def generate_questions(n_questions, databases, config_server):

    databases_paths = [os.path.join("data/databases", db) for db in databases]
    print(databases_paths)
    question_generator=QuestionGenerator(config_server=config_server)
    list_queries, list_answers=[],[]

    for _ in range(n_questions):
        #choose a random db
        db_path = random.choice(databases_paths)

        #choose a random pdf
        pdf_files = [f for f in os.listdir(db_path) if f.lower().endswith(".pdf")]
        if not pdf_files:
            continue 
        pdf_path = os.path.join(db_path, random.choice(pdf_files))

        #choose a random page
        doc = fitz.open(pdf_path)
        page_number = random.randint(0, len(doc) - 1)
        page = doc.load_page(page_number)
        
        #convert this page into bytes
        pix = page.get_pixmap()
        image_bytes = pix.tobytes("png")

        generation = question_generator.generate_question(image_bytes=image_bytes)
        list_queries.append(generation.query)
        list_answers.append(generation.answer)

        doc.close()

    return list_queries, list_answers
    
    


def get_folder_saved_benchmark():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data", "report"))
    folder_bench = []
    for foldername in os.listdir(report_dir):
        folder = os.path.join(report_dir, foldername)
        file_pkl = os.path.join(folder, "results_bench.pkl")
        if os.path.exists(file_pkl):
            folder_bench.append(foldername)
    return folder_bench


def run_indexation_benchmark(
    reset_index, reset_preprocess, databases, report_dir, session_state=None
):
    if session_state is None:
        session_state = st.session_state

    log_file = os.path.join(report_dir, "logs.json")
    if not os.path.exists(log_file):
        data_logs = {
            "indexation": 0.0,
            "answers": 0.0,
            "Arena Battles": 0.0,
            "Ground Truth comparison": 0.0,
            "Context faithfulness": 0.0,
            "context relevance": 0.0,
            "nDCG score": 0.0,
        }
        with open(log_file, "w") as f:
            json.dump(data_logs, f)
    else:
        with open(log_file, "r") as f:
            data_logs = json.load(f)

    rag_agents = []
    rag_names = []
    with st.spinner("**Indexation Running**", show_time=True):
        progress_bar_iterable = [
            rag
            for rag in session_state["benchmark"]["rags"].keys()
            if session_state["benchmark"]["rags"][rag]
        ]
        n = len(progress_bar_iterable)
        indexation_progress_bar = ProgressBar(progress_bar_iterable)
        for i, rag in enumerate(indexation_progress_bar.iterable):
            if session_state["benchmark"]["rags"][rag]:
                rag_agent = get_chat_agent(
                    rag, databases_name=databases, session_state=session_state
                )
                rag_agent.indexation_phase(
                    reset_index=reset_index, reset_preprocess=reset_preprocess
                )
                if reset_preprocess:
                    reset_preprocess = False

                rag_agents.append(rag_agent)
                rag_names.append(rag)
                indexation_progress_bar.update(i)

            data_logs["indexation"] = int(((i + 1) / n) * 100)
            with open(log_file, "w") as f:
                json.dump(data_logs, f)

        indexation_progress_bar.success("Indexation done")
        indexation_progress_bar.clear()
    return rag_agents, rag_names


def get_report_path():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(BASE_DIR, "plot_report_template.tex")

    timestamp = datetime.now()
    timestamp = timestamp.strftime("%m-%d_%H-%M-%S")
    report_dir = os.path.normpath(
        os.path.join(BASE_DIR, "..", "..", "data", "report", timestamp)
    )
    os.makedirs(report_dir, exist_ok=True)

    return report_dir


def generate_only_answers(rag_names, rag_agents, report_dir):

    log_file = os.path.join(report_dir, "logs.json")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            json.dump(
                {
                    "indexation": 0.0,
                    "answers": 0.0,
                    "Arena Battles": 0.0,
                    "Ground Truth comparison": 0.0,
                    "Context faithfulness": 0.0,
                    "context relevance": 0.0,
                },
                f,
            )

    dataframe_preparator = DataFramePreparator(
        rag_agents=rag_agents,
        rags_available=rag_names,
        input_path=os.path.join(
            "data", "queries", st.session_state["benchmark"]["queries_doc_name"]
        ),
    )

    log_file = os.path.join(report_dir, "logs.json")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            json.dump(
                {
                    "indexation": 0.0,
                    "answers": 0.0,
                    "Arena Battles": 0.0,
                    "Ground Truth comparison": 0.0,
                    "Context faithfulness": 0.0,
                    "context relevance": 0.0,
                    "nDCG score": 0.0,
                },
                f,
            )
    dataframe_preparator.run_all_queries(options_generation={"type_generation": "simple_generation"},
                                         log_file=log_file)
    df = dataframe_preparator.get_dataframe()
    df.to_csv(os.path.join(get_report_path(), "bench_df.csv"), index=False)


def generate_only_contexts(rag_names, rag_agents, report_dir):

    log_file = os.path.join(report_dir, "logs.json")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            json.dump(
                {
                    "indexation": 0.0,
                    "answers": 0.0,
                    "Arena Battles": 0.0,
                    "Ground Truth comparison": 0.0,
                    "Context faithfulness": 0.0,
                    "context relevance": 0.0,
                },
                f,
            )

    dataframe_preparator = DataFramePreparator(
        rag_agents=rag_agents,
        rags_available=rag_names,
        input_path=os.path.join(
            "data", "queries", st.session_state["benchmark"]["queries_doc_name"]
        ),
    )
    log_file = os.path.join(report_dir, "logs.json")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            json.dump(
                {
                    "indexation": 0.0,
                    "answers": 0.0,
                    "Arena Battles": 0.0,
                    "Ground Truth comparison": 0.0,
                    "Context faithfulness": 0.0,
                    "context relevance": 0.0,
                    "nDCG score": 0.0,
                },
                f,
            )
    dataframe_preparator.run_all_queries(options_generation={"type_generation": "no_generation"},
                                         log_file=log_file)
    df = dataframe_preparator.get_dataframe()
    df.to_csv(os.path.join(get_report_path(), "contexts_df.csv"), index=False)


def show_already_done_benchmark():
    if st.session_state["benchmark_done"] == "None":
        return

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.normpath(
        os.path.join(
            BASE_DIR, "..", "..", "data", "report", st.session_state["benchmark_done"]
        )
    )
    file_eval_save = os.path.join(report_dir, "results_bench.pkl")
    with open(file_eval_save, "rb") as f:
        results = pickle.load(f)
    plots, impact, energy = show_benchmark(results=results)
    st.session_state["benchmark"]["plots"] = plots
    st.session_state["benchmark"]["report_path"] = report_dir
    st.session_state["benchmark_database"] = results["databases"]


def show_benchmark(results, session_state=None):
    if session_state is None:
        session_state = st.session_state

    (
        session_state["benchmark"]["arena_matrix"],
        session_state["benchmark"]["ground_truth"],
        session_state["benchmark"]["context_faithfulness"],
        session_state["benchmark"]["context_relevance"],
        session_state["benchmark"]["ndcg_score"],
    ) = results["evals"]
    session_state["benchmark"]["ground_truth"] = results["ground_truth_scores"]
    session_state["benchmark"]["arena_matrix"] = results["arena_scores"]
    session_state["benchmark"]["all_queries"] = results["df"]
    session_state["benchmark"]["indexation_tokens"] = results["indexations_tokens"]

    impact = extract_impact(results["df"])
    energy = extract_energy(results["df"])
    time = extract_time(results["df"])
    session_state["benchmark"]["impacts"] = impact
    session_state["benchmark"]["energy"] = energy
    session_state["time"] = time

    arena_graph = arena_graphs(session_state=session_state)
    impacts_graph = impact_graph(session_state=session_state)
    energies_graph = energy_graph(session_state=session_state)


    plots = {
        "token_graph": token_graph(session_state=session_state),
        "ground_truth_graph": ground_truth_graph(session_state=session_state),
        "context_graph": context_graph(session_state=session_state),
        "arena_graphs": arena_graph,
        "report_arena_graph": report_arena_graph(
            arena_graph, session_state=session_state
        ),
        "impact_graph": impacts_graph,
        "energy_graph": energies_graph,
        "time_graph": time_graph(session_state=session_state),
    }
    return plots, impact, energy


def generate_benchmark(
    rag_names,
    rag_agents,
    databases,
    queries_doc_mane,
    config_server,
    models_infos,
    report_dir,
    session_state=None,
):
    if session_state is None:
        session_state = st.session_state

    log_file = os.path.join(report_dir, "logs.json")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            json.dump(
                {
                    "indexation": 0.0,
                    "answers": 0.0,
                    "Arena Battles": 0.0,
                    "Ground Truth comparison": 0.0,
                    "Context faithfulness": 0.0,
                    "context relevance": 0.0,
                    "nDCG score": 0.0,
                },
                f,
            )

    dataframe_preparator = DataFramePreparator(rag_agents=rag_agents,
                                               rags_available=rag_names,
                                               input_path=os.path.join("data", "queries", queries_doc_mane))
    dataframe_preparator.run_all_queries(options_generation={"type_generation": "simple_generation"},
                                         log_file=log_file)

    df = dataframe_preparator.get_dataframe()
    evaluation_agent = AgentEvaluator(dataframe=df,
                                      rags_available=rag_names,
                                      config_server=config_server,
                                      models_infos=models_infos
    )

    evals = evaluation_agent.get_evals(log_file=log_file)

    results_bench = {
        "df": df,
        "evals": evals,
        "ground_truth_scores": evaluation_agent.ground_truth_comparator.all_scores_dict,
        "arena_scores": evaluation_agent.arena.all_scores_dict,
        "indexations_tokens": dataframe_preparator.indexation_tokens,
        "databases": databases,
    }

    file_eval_save = os.path.join(report_dir, "results_bench.pkl")
    with open(file_eval_save, "wb") as f:
        pickle.dump(results_bench, f)

    plots, impact, energy = show_benchmark(
        results=results_bench, session_state=session_state
    )

    session_state["benchmark"]["plots"] = plots
    session_state["benchmark"]["report_path"] = evaluation_agent.create_plot_report(
        plots=plots, report_dir=report_dir
    )
    df_to_save = dataframe_preparator.get_dataframe_to_save()
    df_to_save.to_csv(
        os.path.join(session_state["benchmark"]["report_path"], "bench_df.csv"),
        index=False,
    )
    with open(
        os.path.join(session_state["benchmark"]["report_path"], "impact.json"), "w"
    ) as file:
        json.dump(impact, file, indent=4)
    with open(
        os.path.join(session_state["benchmark"]["report_path"], "energy.json"), "w"
    ) as file:
        json.dump(energy, file, indent=4)


def display_plot():
    st.plotly_chart(st.session_state["benchmark"]["plots"]["token_graph"])
    st.plotly_chart(st.session_state["benchmark"]["plots"]["context_graph"])
    st.plotly_chart(st.session_state["benchmark"]["plots"]["ground_truth_graph"])
    st.plotly_chart(st.session_state["benchmark"]["plots"]["time_graph"])
    if st.session_state["benchmark"]["plots"]["impact_graph"] is not None:
        st.plotly_chart(st.session_state["benchmark"]["plots"]["impact_graph"])
    if st.session_state["benchmark"]["plots"]["energy_graph"] is not None:
        st.plotly_chart(st.session_state["benchmark"]["plots"]["energy_graph"])


def token_graph(session_state=None):
    if session_state is None:
        session_state = st.session_state

    all_queries = session_state["benchmark"]["all_queries"]
    indexation_tokens = session_state["benchmark"]["indexation_tokens"]
    tokens = {}
    for rag in all_queries.columns[2:]:
        tokens[rag] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "embedding_tokens": indexation_tokens[rag]["embedding_tokens"],
            "indexation_input_tokens": indexation_tokens[rag]["input_tokens"],
            "indexation_output_tokens": indexation_tokens[rag]["output_tokens"],
        }

        for query in all_queries[rag]:
            tokens[rag]["input_tokens"] += query["INPUT_TOKENS"]
            tokens[rag]["output_tokens"] += query["OUTPUT_TOKENS"]
    ticksval = []
    data = []
    for rag in tokens.keys():
        ticksval.append(rag)
        data.append(
            {
                "RAG Method": rag,
                "Token Type": "Query Input Tokens",
                "Nb Tokens": tokens[rag]["input_tokens"],
            }
        )
        data.append(
            {
                "RAG Method": rag,
                "Token Type": "Query Output Tokens",
                "Nb Tokens": tokens[rag]["output_tokens"],
            }
        )
        data.append(
            {
                "RAG Method": rag,
                "Token Type": "Embedding Tokens",
                "Nb Tokens": tokens[rag]["embedding_tokens"],
            }
        )
        data.append(
            {
                "RAG Method": rag,
                "Token Type": "Indexation Input Tokens",
                "Nb Tokens": tokens[rag]["indexation_input_tokens"],
            }
        )
        data.append(
            {
                "RAG Method": rag,
                "Token Type": "Indexation Output Tokens",
                "Nb Tokens": tokens[rag]["indexation_output_tokens"],
            }
        )
    tickstext = [session_state["all_rags"][tick] for tick in ticksval]
    fig = px.bar(
        data,
        x="RAG Method",
        y="Nb Tokens",
        color="Token Type",
        barmode="stack",
        title="Token Consumption",
        labels={"Nb Tokens": "Nb Tokens", "RAG Method": "RAG Method"},
        color_discrete_sequence=color_discrete_sequence,
    )

    fig.update_layout(
        title=dict(text="Token Consumption", x=0.5, xanchor="center"),
        yaxis={"title": {"text": "Nb Tokens"}},
        xaxis={"title": {"text": ""}, "tickvals": ticksval, "ticktext": tickstext},
        legend={"title": ""},
        legend_traceorder="reversed",
    )

    return fig


def context_graph(session_state=None):
    if session_state is None:
        session_state = st.session_state

    faithfulness = session_state["benchmark"]["context_faithfulness"]
    relevance = session_state["benchmark"]["context_relevance"]
    ndcg = session_state["benchmark"]["ndcg_score"]

    ticksval = []
    data = []
    for rag in relevance.keys():
        ticksval.append(rag)
        data.append(
            {"RAG Method": rag, "Score": relevance[rag], "Metric": "Context Relevance"}
        )
    for rag in faithfulness.keys():
        data.append(
            {
                "RAG Method": rag,
                "Score": faithfulness[rag],
                "Metric": "Context Faithfulness",
            }
        )
    for rag in ndcg.keys():
        data.append(
            {
                "RAG Method": rag,
                "Score": ndcg[rag],
                "Metric": "Context nDCG Score",
            }
        )
    tickstext = [session_state["all_rags"][tick] for tick in ticksval]
    fig = px.bar(
        data,
        x="Score",
        y="RAG Method",
        color="Metric",
        barmode="group",
        title="Context Analysis",
        labels={"Score": "Score", "RAG Method": "RAG Method"},
        orientation="h",
        color_discrete_sequence=color_discrete_sequence,
    )

    fig.update_layout(
        title=dict(text="Context Analysis", x=0.5, xanchor="center"),
        xaxis={"title": {"text": "Score"}},
        yaxis={"title": {"text": ""}, "tickvals": ticksval, "ticktext": tickstext},
        legend_traceorder="reversed",
        legend={"title": ""},
        margin={"l": 150},
    )

    return fig


def ground_truth_graph(session_state=None):
    if session_state is None:
        session_state = st.session_state

    ground_truth = session_state["benchmark"]["ground_truth"]

    data = []
    ticksval = []
    for rag in ground_truth.keys():
        ticksval.append(rag)
        for metric in ground_truth[rag]:
            data.append(
                {
                    "RAG Method": rag,
                    "Score": ground_truth[rag][metric],
                    "Metric": metric,
                }
            )

    fig = px.bar(
        data,
        x="Score",
        y="RAG Method",
        color="Metric",
        barmode="group",
        title="Ground Truth Analysis",
        labels={"Score": "Score", "RAG Method": "RAG Method"},
        orientation="h",
        color_discrete_sequence=color_discrete_sequence,
    )
    tickstext = [session_state["all_rags"][tick] for tick in ticksval]
    fig.update_layout(
        title=dict(text="Ground Truth Analysis", x=0.5, xanchor="center"),
        xaxis={"title": {"text": "Score"}},
        yaxis={"title": {"text": ""}, "tickvals": ticksval, "ticktext": tickstext},
        legend={"title": {"text": ""}},
        legend_traceorder="reversed",
        margin={"l": 150},
    )
    return fig


def arena_graphs(session_state=None):
    if session_state is None:
        session_state = st.session_state

    figures = {}
    arena_matrix = session_state["benchmark"]["arena_matrix"]
    for match in arena_matrix.keys():
        mid = match.find("_v_")
        rag1 = match[:mid]
        rag2 = match[mid + 3 :]
        data = []
        for metric in arena_matrix[match].keys():
            data.append(
                {
                    "Metric": metric,
                    "RAG": session_state["all_rags"][rag1],
                    "Score": arena_matrix[match][metric][0],
                }
            )
            data.append(
                {
                    "Metric": metric,
                    "RAG": session_state["all_rags"][rag2],
                    "Score": arena_matrix[match][metric][1],
                }
            )

        fig = px.bar(
            data,
            x="Score",
            y="Metric",
            color="RAG",
            barmode="stack",
            labels={"Score": "", "Metric": ""},
            color_discrete_sequence=color_discrete_sequence,
        )

        fig.update_layout(xaxis=dict(range=[0, 100]), margin={"l": 100})
        figures[match] = fig
    return figures


def report_arena_graph(arena_graphs, session_state=None):
    if session_state is None:
        session_state = st.session_state

    n_cols = max(1, int((-1 + sqrt(1 + 8 * len(arena_graphs))) / 2))

    rag_list = []
    session_state["benchmark"]["matches"] = list(arena_graphs.keys())
    for match in list(arena_graphs.keys())[0:n_cols]:
        mid = match.find("_v_")
        rag_b = match[mid + 3 :]
        rag_list.append(session_state["all_rags"][rag_b])

    while len(rag_list) < n_cols**2:
        rag_list.append("")

    fig = make_subplots(rows=n_cols, cols=n_cols, subplot_titles=rag_list)
    row = 0
    col = 1
    prev_rag_a = ""
    y_titles = []
    yaxis = {}

    for match, figure in arena_graphs.items():
        mid = match.find("_v_")
        rag_a = match[:mid]
        rag_b = match[mid + 3 :]

        if rag_a != prev_rag_a:
            row += 1
            col = 1
            y_titles.append(session_state["all_rags"][rag_a])

        for trace in figure.data:
            fig.add_trace(trace, row=row, col=col)
        col += 1
        prev_rag_a = rag_a

    for i in range(n_cols):
        for j in range(n_cols):
            title_sufix = f"{n_cols*i+j+1}"
            graph_title = y_titles[i] if j == 0 else " "
            yaxis["yaxis" + f"{title_sufix}"] = {
                "showticklabels": False,
                "title": graph_title,
            }

    fig.update_layout(
        height=n_cols * 250,
        width=n_cols * 350,
        showlegend=False,
        barmode="stack",
        **yaxis,
        font={"size": 11, "color": "black", "family": "Arial, sans-serif"},
    )
    return fig


def match_name_cleaner(match_name):
    mid = match_name.find("_v_")

    rag_a = match_name[:mid]
    rag_b = match_name[mid + 3 :]
    try:
        new_rag_name = f"{st.session_state['all_rags'][rag_a]} - {st.session_state['all_rags'][rag_b]}"
    except Exception:
        new_rag_name = f"{rag_a} - {rag_b}"

    return new_rag_name


def extract_impact(df: pd.DataFrame):
    list_rags = list(df.columns)[2:]
    impacts = {}
    for rag in list_rags:
        impact = [0, 0, "gCO2eq"]
        for query in df[rag]:
            if type(query) is not dict:
                q = eval(query)
            else:
                q = query
            impact[0] += q["IMPACTS"][0] * 1000
            impact[1] += q["IMPACTS"][1] * 1000
        impacts[rag] = impact
    return impacts


def extract_energy(df: pd.DataFrame):
    list_rags = list(df.columns)[2:]
    energies = {}
    for rag in list_rags:
        energy = [0, 0, "wH"]
        for query in df[rag]:
            if type(query) is not dict:
                q = eval(query)
            else:
                q = query
            energy[0] += q["ENERGY"][0] * 1000
            energy[1] += q["ENERGY"][1] * 1000
        energies[rag] = energy
    return energies


def extract_time(df: pd.DataFrame):
    list_rags = list(df.columns)[2:]
    time = {}
    for rag in list_rags:
        time[rag] = df[rag][0]["TIME"]
    return time


def impact_graph(session_state=None):
    if session_state is None:
        session_state = st.session_state

    impacts = session_state["benchmark"]["impacts"]
    data = []
    for rag in impacts.keys():
        data.append(
            {
                "RAG Method": session_state["all_rags"][rag],
                "center": (impacts[rag][0] + impacts[rag][1]) / 2,
                "error": (impacts[rag][0] - impacts[rag][1]) / 2,
            }
        )

    fig = px.scatter(
        data,
        x="RAG Method",
        y="center",
        error_y="error",
        color="RAG Method",
        color_discrete_sequence=color_discrete_sequence,
    )
    fig.update_traces(
        marker=dict(size=16, symbol="square", line=dict(width=1, color="black"))
    )
    fig.update_layout(
        yaxis_title="Greenhouse gas emissions (gCO2eq)",
        title=dict(
            text="Greenhouse gas emissions estimations", x=0.5, xanchor="center"
        ),
    )
    return fig


def energy_graph(session_state=None):
    if session_state is None:
        session_state = st.session_state

    energies = session_state["benchmark"]["energy"]
    data = []
    for rag in energies.keys():
        data.append(
            {
                "RAG Method": session_state["all_rags"][rag],
                "center": (energies[rag][0] + energies[rag][1]) / 2,
                "error": (energies[rag][0] - energies[rag][1]) / 2,
            }
        )

    fig = px.scatter(
        data,
        x="RAG Method",
        y="center",
        error_y="error",
        color="RAG Method",
        color_discrete_sequence=color_discrete_sequence,
    )
    fig.update_traces(
        marker=dict(size=16, symbol="square", line=dict(width=1, color="black"))
    )
    fig.update_layout(
        yaxis_title="Power used (kWh)",
        title=dict(text="Energy consumption estimations", x=0.5, xanchor="center"),
    )
    return fig


def time_graph(session_state=None):
    if session_state is None:
        session_state = st.session_state

    raw_data = session_state["time"]
    data = []
    for rag in raw_data:
        rag_data = {
            "RAG Method": session_state["all_rags"][rag],
            "Answering Time": raw_data[rag],
        }
        data.append(rag_data)

    fig = px.bar(
        data,
        x="RAG Method",
        y="Answering Time",
        color="RAG Method",
        color_discrete_sequence=color_discrete_sequence,
    )

    fig.update_layout(
        title=dict(text="Answering Time comparison", x=0.5, xanchor="center"),
        legend={"title": {"text": ""}},
        yaxis={"title": {"text": "Answering Time (s)"}},
    )

    return fig


def clean_bench_df():
    answers = pd.read_csv(
        st.session_state["benchmark"]["report_path"] + "/bench_df.csv"
    )
    return answers.to_excel(
        excel_writer=st.session_state["benchmark"]["report_path"] + "/answers.xlsx",
        sheet_name="answers",
        engine="openpyxl",
    )
