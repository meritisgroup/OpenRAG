import streamlit as st
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

from streamlit_.utils.chat_funcs import get_chat_agent
from backend.utils.progress import ProgressBar


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

def run_indexation_benchmark(reset_index):
    rag_agents = []
    rag_names = []
    with st.spinner("**Indexation Running**", show_time=True):
        progress_bar_iterable = [
            rag
            for rag in st.session_state["benchmark"]["rags"].keys()
            if st.session_state["benchmark"]["rags"][rag]
        ]
        indexation_progress_bar = ProgressBar(progress_bar_iterable)
        for i, rag in enumerate(indexation_progress_bar.iterable):
            if st.session_state["benchmark"]["rags"][rag]:
                rag_agent = get_chat_agent(rag,
                                           databases_name=[st.session_state['benchmark_database']])
                rag_agent.indexation_phase(reset_index=reset_index)
                rag_agents.append(rag_agent)
                rag_names.append(rag)
                indexation_progress_bar.update(i)
        indexation_progress_bar.success("Indexation done")
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

def generate_answers(rag_names, rag_agents):
    dataframe_preparator = DataFramePreparator(
        rag_agents=rag_agents,
        rags_available=rag_names,
        input_path=os.path.join("data", "queries", 
                                st.session_state["benchmark"]["queries_doc_name"]),
    )
    dataframe_preparator.run_all_queries(options_generation={"type_generation": "simple_generation"})
    df = dataframe_preparator.get_dataframe()
    df.to_csv(os.path.join(get_report_path(), "bench_df.csv"), index=False)
       
def generate_contexts(rag_names, rag_agents):
    dataframe_preparator = DataFramePreparator(
        rag_agents=rag_agents,
        rags_available=rag_names,
        input_path=os.path.join("data", "queries",
                                st.session_state["benchmark"]["queries_doc_name"]),
    )
    dataframe_preparator.run_all_queries(options_generation={"type_generation": "no_generation"})
    df = dataframe_preparator.get_dataframe()
    df.to_csv(os.path.join(get_report_path(), "contexts_df.csv"), index=False)


def generate_benchmark(rag_names, rag_agents):

    dataframe_preparator = DataFramePreparator(
        rag_agents=rag_agents,
        rags_available=rag_names,
        input_path=os.path.join("data", "queries",
                                 st.session_state["benchmark"]["queries_doc_name"])
    )
    dataframe_preparator.run_all_queries(options_generation={"type_generation": "simple_generation"})

    df = dataframe_preparator.get_dataframe()
    evaluation_agent = AgentEvaluator(
        dataframe=df,
        rags_available=rag_names,
        config_server=st.session_state["config_server"],
    )

    evals = evaluation_agent.get_evals()
    (
        st.session_state["benchmark"]["arena_matrix"],
        st.session_state["benchmark"]["ground_truth"],
        st.session_state["benchmark"]["context_faithfulness"],
        st.session_state["benchmark"]["context_relevance"],
    ) = evals
    st.session_state["benchmark"]["ground_truth"] = evaluation_agent.ground_truth_comparator.all_scores_dict
    st.session_state["benchmark"]["arena_matrix"] = evaluation_agent.arena.all_scores_dict
    st.session_state["benchmark"]["all_queries"] = dataframe_preparator.dataframe
    st.session_state["benchmark"]["indexation_tokens"] = dataframe_preparator.indexation_tokens

    impact = extract_impact(df)
    energy = extract_energy(df)
    time = extract_time(df)
    st.session_state["benchmark"]["impacts"] = impact
    st.session_state["benchmark"]["energy"] = energy
    st.session_state["time"] = time
    
    arena_graph = arena_graphs()
    if st.session_state["config_server"]["params_host_llm"]["type"] in ["openai", "mistral"]:
        impacts_graph = impact_graph()
        energies_graph = energy_graph()
    
    else:
        impacts_graph = None
        energies_graph = None

    plots = {
        "token_graph": token_graph(),
        "ground_truth_graph": ground_truth_graph(),
        "context_graph": context_graph(),
        "arena_graphs": arena_graph,
        "report_arena_graph": report_arena_graph(arena_graph),
        "impact_graph" : impacts_graph,
        "energy_graph" : energies_graph,
        "time_graph" : time_graph()
    }
    st.session_state["benchmark"]["plots"] = plots
    st.session_state["benchmark"]["report_path"] = evaluation_agent.create_plot_report(plots=plots,
                                                                                       report_dir=get_report_path())
    df.to_csv(os.path.join(st.session_state["benchmark"]["report_path"], "bench_df.csv"), index=False)
    with open(os.path.join(st.session_state["benchmark"]["report_path"], "impact.json"), "w") as file:
        json.dump(impact,file,indent=4)
    with open(os.path.join(st.session_state["benchmark"]["report_path"], "energy.json"), "w") as file:
        json.dump(energy,file,indent=4)



def display_plot():
    st.plotly_chart(st.session_state["benchmark"]["plots"]["token_graph"])
    st.plotly_chart(st.session_state["benchmark"]["plots"]["context_graph"])
    st.plotly_chart(st.session_state["benchmark"]["plots"]["ground_truth_graph"])
    st.plotly_chart(st.session_state["benchmark"]["plots"]["time_graph"])
    if st.session_state["benchmark"]["plots"]["impact_graph"] is not None:
        st.plotly_chart(st.session_state["benchmark"]["plots"]["impact_graph"])
    if st.session_state["benchmark"]["plots"]["energy_graph"] is not None:
        st.plotly_chart(st.session_state["benchmark"]["plots"]["energy_graph"])


def token_graph():
    all_queries = st.session_state["benchmark"]["all_queries"]
    indexation_tokens = st.session_state["benchmark"]["indexation_tokens"]
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
    host = st.session_state["config_server"]["params_host_llm"]["type"]
    tickstext = [st.session_state["all_rags"][host][tick] for tick in ticksval]
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


def context_graph():
    faithfulness = st.session_state["benchmark"]["context_faithfulness"]
    relevance = st.session_state["benchmark"]["context_relevance"]

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
    host = st.session_state["config_server"]["params_host_llm"]["type"]
    tickstext = [st.session_state["all_rags"][host][tick] for tick in ticksval]
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


def ground_truth_graph():
    ground_truth = st.session_state["benchmark"]["ground_truth"]

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
    host = st.session_state["config_server"]["params_host_llm"]["type"]
    tickstext = [st.session_state["all_rags"][host][tick] for tick in ticksval]
    fig.update_layout(
        title=dict(text="Ground Truth Analysis", x=0.5, xanchor="center"),
        xaxis={"title": {"text": "Score"}},
        yaxis={"title": {"text": ""}, "tickvals": ticksval, "ticktext": tickstext},
        legend={"title": {"text": ""}},
        legend_traceorder="reversed",
        margin={"l": 150},
    )
    return fig


def arena_graphs():
    figures = {}
    arena_matrix = st.session_state["benchmark"]["arena_matrix"]
    for match in arena_matrix.keys():
        mid = match.find("_v_")
        rag1 = match[:mid]
        rag2 = match[mid + 3 :]
        data = []
        host = st.session_state["config_server"]["params_host_llm"]["type"]
        for metric in arena_matrix[match].keys():
            data.append(
                {
                    "Metric": metric,
                    "RAG": st.session_state["all_rags"][host][rag1],
                    "Score": arena_matrix[match][metric][0],
                }
            )
            data.append(
                {
                    "Metric": metric,
                    "RAG": st.session_state["all_rags"][host][rag2],
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


def report_arena_graph(arena_graphs):
    n_cols = max(1, int((-1 + sqrt(1 + 8 * len(arena_graphs))) / 2))

    rag_list = []
    host = st.session_state["config_server"]["params_host_llm"]["type"]
    st.session_state["benchmark"]["matches"] = list(arena_graphs.keys())
    for match in list(arena_graphs.keys())[0:n_cols]:
        mid = match.find("_v_")
        rag_b = match[mid + 3 :]
        rag_list.append(st.session_state["all_rags"][host][rag_b])

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
            y_titles.append(st.session_state["all_rags"][host][rag_a])

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
    rag_b = match_name[mid+3:]
    host  = st.session_state["config_server"]["params_host_llm"]["type"]
    try:
        new_rag_name = f"{st.session_state['all_rags'][host][rag_a]} - {st.session_state['all_rags'][host][rag_b]}"
    except Exception:
        new_rag_name = f"{rag_a} - {rag_b}"

    return new_rag_name


def extract_impact(df: pd.DataFrame):
    list_rags = list(df.columns)[2:]
    impacts = {}
    for rag in list_rags:
        impact = [0,0,"gCO2eq"]
        for query in df[rag]:
            if type(query) is not dict:
                q = eval(query)
            else:
                q = query
            impact[0] += q["IMPACTS"][0]*1000
            impact[1] += q["IMPACTS"][1]*1000
        impacts[rag] = impact
    return impacts

def extract_energy(df: pd.DataFrame):
    list_rags = list(df.columns)[2:]
    energies = {}
    for rag in list_rags:
        energy = [0,0,"wH"]
        for query in df[rag]:
            if type(query) is not dict:
                q = eval(query)
            else:
                q = query
            energy[0] += q["ENERGY"][0]*1000
            energy[1] += q["ENERGY"][1]*1000
        energies[rag] = energy
    return energies

def extract_time(df : pd.DataFrame):
    list_rags = list(df.columns)[2:]
    time = {}
    for rag in list_rags:
        time[rag] = df[rag][0]['TIME']
    return time

def impact_graph():
    impacts = st.session_state["benchmark"]["impacts"]
    host = st.session_state["config_server"]["params_host_llm"]["type"]
    data = []
    for rag in impacts.keys():
        data.append(
            {
                "RAG Method" : st.session_state["all_rags"][host][rag],
                "center" : (impacts[rag][0] + impacts[rag][1]) / 2,
                "error" : (impacts[rag][0] - impacts[rag][1]) / 2
            }
        )
    
    fig = px.scatter(
        data,
        x="RAG Method",
        y="center",
        error_y="error",
        color="RAG Method",
        color_discrete_sequence=color_discrete_sequence
    )
    fig.update_traces(marker=dict(size=16, symbol="square", line=dict(width=1, color="black")))
    fig.update_layout(yaxis_title="Greenhouse gas emissions (gCO2eq)",
                      title=dict(text="Greenhouse gas emissions estimations", x=0.5, xanchor="center"))
    return fig

def energy_graph():
    energies = st.session_state["benchmark"]["energy"]
    host = st.session_state["config_server"]["params_host_llm"]["type"]
    data = []
    for rag in energies.keys():
        data.append(
            {
                "RAG Method" : st.session_state["all_rags"][host][rag],
                "center" : (energies[rag][0] + energies[rag][1]) / 2,
                "error" : (energies[rag][0] - energies[rag][1]) / 2
            }
        )
    
    fig = px.scatter(
        data,
        x="RAG Method",
        y="center",
        error_y="error",
        color="RAG Method",
        color_discrete_sequence=color_discrete_sequence
    )
    fig.update_traces(marker=dict(size=16, symbol="square", line=dict(width=1, color="black")))
    fig.update_layout(yaxis_title="Power used (kWh)",
                      title=dict(text="Energy consumption estimations", x=0.5, xanchor="center"))
    return fig

def time_graph():
    raw_data = st.session_state["time"]
    data = []
    host = st.session_state["config_server"]["params_host_llm"]["type"]
    for rag in raw_data:
        rag_data = {"RAG Method" : st.session_state["all_rags"][host][rag],
                    "Answering Time" : raw_data[rag]}
        data.append(rag_data)

    fig = px.bar(
        data,
        x= "RAG Method",
        y="Answering Time",
        color="RAG Method",
        color_discrete_sequence=color_discrete_sequence

    )

    fig.update_layout(
            title=dict(text="Answering Time comparison", x=0.5, xanchor="center"),
            legend={"title": {"text": ""}},
            yaxis={"title": {"text": "Answering Time (s)"}}
        )

    return fig

def clean_bench_df():
    answers = pd.read_csv(st.session_state["benchmark"]["report_path"] + "/bench_df.csv")
    rags = list(answers.columns[2:])
    for rag in rags:
        for i,query in enumerate(answers[rag]):
            query = eval(query)
            answers.loc[i, rag] = query["ANSWER"]
    return answers.to_excel(excel_writer=st.session_state["benchmark"]["report_path"] +"/answers.xlsx", sheet_name="answers", engine="openpyxl")