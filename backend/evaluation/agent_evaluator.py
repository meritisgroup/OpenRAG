from .comparators import (
    ArenaBattle,
    GroundTruthComparator,
    ContextFaithfulnessComparator,
    ContextRelevanceComparator,
)
from sqlalchemy import func
import numpy as np
from backend.database.rag_classes import Document, Tokens
import time
from ..utils.agent import get_Agent
from ..base_classes import RagAgent
from datetime import datetime
import pandas as pd
import os
import subprocess
from backend.methods.graph_rag.agent import GraphRagAgent
from ..utils.progress import ProgressBar


class AgentEvaluator:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        rags_available: list[str],
        config_server: dict,
    ):
        """This Agent takes a Dataframe as input and run all evaluations on it.
        The dataframe must includes a "QUERY" and a "GROUND_TRUTH" columns along with all the rags answers from the rag we are evaluating.
        In every rag column, and for each query, we need a dictionary with "answer" and "context" keys to evaluate both of them relatively to each rag
        """
        self.agent = get_Agent(config_server)

        self.dataframe = dataframe

        self.gt_dataframe = dataframe[dataframe["GROUND_TRUTH"].notna()]

        self.arena = ArenaBattle(dataframe=self.gt_dataframe, agent=self.agent)
        self.ground_truth_comparator = GroundTruthComparator(
            dataframe=self.gt_dataframe, agent=self.agent
        )
        self.context_faithfulness_comparator = ContextFaithfulnessComparator(
            dataframe=self.gt_dataframe, agent=self.agent
        )
        self.context_relevance_comparator = ContextRelevanceComparator(
            dataframe=self.dataframe, agent=self.agent
        )
        self.rags_available = rags_available

    def get_evals(self):
        print("Running Arena Battles ...")
        arena_matrix = self.arena.run_battles_scores()
        print("Arena battles done  - ✅")
        print("Running Ground Truth comparison ...")
        ground_truth_evaluations = self.ground_truth_comparator.run_evaluations()
        print("Ground Truth comparison done  - ✅")
        print("Running context faithfulness ...")
        context_faithfulness_evaluations = (
            self.context_faithfulness_comparator.run_evaluations()
        )
        print("Context faithfulness done  - ✅")
        print("Running context relevance ...")
        context_relevance_evaluations = (
            self.context_relevance_comparator.run_evaluations()
        )
        print("Context relevance done  - ✅")
        return (
            arena_matrix,
            ground_truth_evaluations,
            context_faithfulness_evaluations,
            context_relevance_evaluations,
        )

    def create_plot_report(self, plots, report_dir) -> str:

        plots["token_graph"].write_image(report_dir + "/tokens.png", format="png")
        plots["ground_truth_graph"].write_image(
            report_dir + "/ground_truth.png", format="png"
        )
        plots["context_graph"].write_image(report_dir + "/context.png", format="png")
        plots["time_graph"].write_image(report_dir + "/time_graph.png", format="png")
        for match, fig in plots["arena_graphs"].items():
            fig.write_image(report_dir + f"/{match}.png", format="png")
        plots["report_arena_graph"].write_image(
            report_dir + "/report_arena_graph.png", format="png"
        )

        for file in os.listdir(report_dir):
            if "_v_" in file:
                example_arena = file
                break

        with open(template_path, "r", encoding="utf-8") as report_template:
            content = report_template.read()
            final_report = (
                content.replace("{token_graph_path}", report_dir + "/tokens.png")
                .replace("{gt_graph_path}", report_dir + "/ground_truth.png")
                .replace("{context_graph_path}", report_dir + "/context.png")
                .replace("{example_arena_graph}", report_dir + "/" + example_arena)
                .replace("{report_arena_graph}", report_dir + "/report_arena_graph.png")
                .replace("{time_graph}", report_dir + "/time_graph.png")
            )
        if plots["impact_graph"] is not None:
            plots["impact_graph"].write_image(
                report_dir + "/impact_graph.png", format="png"
            )
            final_report = self.add_impact_sequence(final_report)

        if plots["energy_graph"] is not None:
            plots["energy_graph"].write_image(
                report_dir + "/energy_graph.png", format="png"
            )
            final_report = self.add_energy_sequence(final_report)

        tex_filename = "plot_report.tex"
        tex_path = os.path.join(report_dir, tex_filename)
        with open(tex_path, "w+") as f:
            f.write(final_report)
        self.tex_to_pdf(tex_path)
        return report_dir

    def tex_to_pdf(self, tex_file_path) -> None:
        if not os.path.exists(tex_file_path):
            print(f"File not found: {tex_file_path}")
            return

        tex_dir = os.path.dirname(os.path.abspath(tex_file_path))
        tex_filename = os.path.basename(tex_file_path)

        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_filename],
                cwd=tex_dir,
                check=True,
            )
            print(f"PDF created successfully in: {tex_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error during PDF generation : {e}")

    def add_impact_sequence(self, template: str) -> str:
        end = template.find("\\end{document}")
        new_template = template[:end]
        new_template += """
                        \\section{Greenhouse gas emissions}
                    Here is an estimation of how much greenhouse gas each RAG has emitted while performing the benchmark, please note that the indexation is not taken into account. Each bar represents an interval of gCO2eq and has been computed using Ecologits library.  

                    \\begin{figure}[H]
                        \\centering
                        \\includegraphics[width = 14cm]{{impact_graph}}
                    \\end{figure}

                    \\end{document}
                        """
        return new_template

    def add_energy_sequence(self, template: str) -> str:
        end = template.find("\\end{document}")
        new_template = template[:end]
        new_template += """
                        \\section{Power consumption}
                    Here is an estimation of how much power each RAG has used while performing the benchmark, please note that the indexation is not taken into account. Each bar represents an interval of kWh and has been computed using Ecologits library.  

                    \\begin{figure}[H]
                        \\centering
                        \\includegraphics[width = 14cm]{{energy_graph}}
                    \\end{figure}

                    \\end{document}
                        """
        return new_template


class DataFramePreparator:

    def __init__(
        self, rag_agents: list[RagAgent], rags_available: list[str], input_path: str
    ):
        self.rag_agents = rag_agents
        self.rags_available = rags_available
        self.input_path = input_path

        self.queries, self.ground_truths = self.get_queries()
        self.column_names = ["QUERIES", "GROUND_TRUTH"] + rags_available

        data = {
            self.column_names[0]: self.queries,
            self.column_names[1]: self.ground_truths,
        }
        self.dataframe = pd.DataFrame(data, columns=self.column_names)
        self.indexation_tokens = {}
        

    def get_dataframe(self) -> pd.DataFrame:
        return self.dataframe

    def get_queries(self) -> list[str]:
        queries = []
        ground_truths = []
        df = pd.read_excel(io=self.input_path, engine="openpyxl")
        queries = df["query"].tolist()
        ground_truths = [
            ans if pd.notna(ans) and ans != "" else None for ans in df["answer"]
        ]

        return queries, ground_truths

    def run_all_queries(self, options_generation = None) -> None:

        progress_bar = ProgressBar(
            zip(self.rags_available, self.rag_agents),
            total=len(self.rags_available),
            desc="Generating RAG answers",
        )
        indexation_tokens = self.indexation_tokens
        n = len(self.rags_available)
        for i, (rag_available, rag_agent) in enumerate(progress_bar.iterable):
            progress_bar.update(
                i - 1,
                text=f"Generating RAG Answers for {rag_available} rag ({i+1}/{n})",
            )
            indexation_tokens[rag_available] = {}
            if type(rag_agent) is GraphRagAgent:
                indexation_tokens[rag_available]["embedding_tokens"] = (
                    np.sum(rag_agent.data_manager.query(func.sum(Tokens.embedding_tokens)))
                )
                indexation_tokens[rag_available]["input_tokens"] = np.sum(rag_agent.data_manager.query(
                    func.sum(Tokens.input_tokens)
                ))
                indexation_tokens[rag_available]["output_tokens"] = np.sum(rag_agent.data_manager.query(
                    func.sum(Tokens.output_tokens)
                ))
            else:
                indexation_tokens[rag_available]["embedding_tokens"] = (
                    np.sum(rag_agent.data_manager.query(func.sum(Document.embedding_tokens)))
                )
                indexation_tokens[rag_available]["input_tokens"] = np.sum(rag_agent.data_manager.query(
                    func.sum(Document.input_tokens)
                ))
                indexation_tokens[rag_available]["output_tokens"] = np.sum(rag_agent.data_manager.query(
                    func.sum(Document.output_tokens)
                ))
            start_time = time.time()
            rag_results = rag_agent.generate_answers(self.queries, 
                                                     rag_agent.nb_chunks,
                                                     options_generation=options_generation)
            end_time = time.time()
            answer_time = end_time - start_time
            answers = [rag_result["answer"] for rag_result in rag_results]
            contexts = [rag_result["context"] for rag_result in rag_results]
            nb_input_tokens = [
                rag_result["nb_input_tokens"] for rag_result in rag_results
            ]
            nb_output_tokens = [
                rag_result["nb_output_tokens"] for rag_result in rag_results
            ]
            impacts = [rag_result["impacts"] for rag_result in rag_results]

            energies = [rag_result["energy"] for rag_result in rag_results]
            self.dataframe[rag_available] = [
                dict(
                    ANSWER=answer,
                    CONTEXT=context,
                    INPUT_TOKENS=nb_input_token,
                    OUTPUT_TOKENS=nb_output_token,
                    IMPACTS=impact,
                    ENERGY=energy,
                    TIME=answer_time,
                )
                for answer, context, nb_input_token, nb_output_token, impact, energy in zip(
                    answers,
                    contexts,
                    nb_input_tokens,
                    nb_output_tokens,
                    impacts,
                    energies,
                )
            ]

        progress_bar.success("Answers ready for evaluation")
