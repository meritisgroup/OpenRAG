from ..utils.agent import Agent
from .end_to_end_evaluators import MetricComparaison, GroundTruthComparison
from .context_evaluators import ContextFaithfulnessEvaluator, ContextRelevanceEvaluator, nDCGEvaluator
from .prompts import PROMPTS
from ..utils.progress import ProgressBar
from ..methods.naive_rag.indexation import concat_chunks
from backend.database.rag_classes import Chunk
from ..database.rag_classes import Chunk

from ..methods.naive_rag.indexation import concat_chunks

import pandas as pd
import json
import numpy as np


class ArenaBattle:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        agent: Agent,
        model: str,
        eval_number=1,
        max_attempts=5,
        batch_size=10,
    ):
        """
        An ArenaBattle takes a dataframe with a column of queries and for each RAG method,
        a column containing the corresponding retrieved answers.

        It evaluates different RAG methods based on predefined metrics and generates a score matrix.
        """
        self.dataframe = dataframe
        self.queries = dataframe["QUERIES"]
        self.ground_truths = self.dataframe["GROUND_TRUTH"]
        self.rag_list = [
            col
            for col in dataframe.columns
            if col != "QUERIES" and col != "GROUND_TRUTH"
        ]
        self.agent = agent
        self.model = model
        self.temperature = agent.temperature

        self.metric_evaluator = MetricComparaison(
            agent=self.agent, max_attemps=max_attempts, batch_size=batch_size,
            model=self.model
        )
        self.language = self.agent.language
        self.metrics: dict = PROMPTS[self.language]["metrics"]
        self.eval_number = eval_number

    def process_round_scores(self, first_rag: str, second_rag: str):

        first_answers = [item["ANSWER"] for item in self.dataframe[first_rag]]
        second_answers = [item["ANSWER"] for item in self.dataframe[second_rag]]
        final_scores = {metric: (0.0, 0.0) for metric in self.metrics.keys()}

        for eval_idx in range(self.eval_number):

            dict_scores = self.metric_evaluator.run_evaluation_pipeline(
                self.queries, self.ground_truths, first_answers, second_answers
            )
            for metric in self.metrics.keys():
                first_mean, second_mean = final_scores[metric]
                first_result, second_result = dict_scores[metric]

                try:
                    final_scores[metric] = (first_result + first_mean * eval_idx) / (
                        eval_idx + 1
                    ), (second_result + second_mean * eval_idx) / (eval_idx + 1)
                except Exception:
                    final_scores[metric] = (None, None)

        return final_scores

    def run_battles_scores(self, log_file):
        """
        Runs battles between all RAG models and computes score matrices.

        Returns:
            dict: A dictionary of score matrices for each metric.
        """
        num_rags = len(self.rag_list)
        all_scores = {metric: np.zeros((num_rags, num_rags)) for metric in self.metrics}
        all_scores_dict = {}
        total_comparisons = (num_rags * (num_rags - 1)) // 2  # Total pairs to compare
        progress_bar = ProgressBar(
            total=total_comparisons, desc="Arena Battles Progress"
        )
        k = 0
        with open(log_file, "r") as f:
            data_logs = json.load(f)

        for i in range(num_rags):
            for j in range(i + 1, num_rags):

                first_rag = self.rag_list[i]
                second_rag = self.rag_list[j]
                all_scores_dict[f"{first_rag}_v_{second_rag}"] = {
                    metric: [0, 0] for metric in self.metrics
                }

                round_scores = self.process_round_scores(first_rag, second_rag)

                for metric in self.metrics.keys():
                    first_result, second_result = round_scores[metric]
                    all_scores[metric][i][j] = first_result
                    all_scores[metric][j][i] = second_result
                    all_scores_dict[f"{first_rag}_v_{second_rag}"][metric] = [
                        first_result,
                        second_result,
                    ]

                progress_bar.update(
                    k,
                    text=f"Arena Battles Progress ({k+1} / {int(num_rags * int((num_rags - 1))/2)})",
                )
                k += 1
                data_logs["Arena Battles"] = int(
                    ((k + 1) / (int(num_rags * int((num_rags - 1)) / 2))) * 100
                )
                with open(log_file, "w") as f:
                    json.dump(data_logs, f)
        self.all_scores_dict = all_scores_dict
        progress_bar.success("Arena battles completed")

        return all_scores


class GroundTruthComparator:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        agent: Agent,
        model: str,
        eval_number=1,
        max_attempts=5,
        batch_size=10,
    ):

        self.agent = agent
        self.comparator = GroundTruthComparison(agent=agent, 
                                                model=model,
                                                max_attemps=max_attempts,
                                                batch_size=batch_size)
        self.language = self.agent.language
        self.metrics: dict = PROMPTS[self.language]["gt_metrics"]

        self.dataframe = dataframe
        self.queries = dataframe["QUERIES"]
        self.ground_truth = dataframe["GROUND_TRUTH"]

        self.rag_list = [
            col
            for col in dataframe.columns
            if (col != "QUERIES" and col != "GROUND_TRUTH")
        ]
        self.eval_number = eval_number


    def process_evaluation(self, rag_answers: list[str], rag_name:str):

        final_scores = {metric: 0.0 for metric in self.metrics.keys()}
        final_evaluation = {metric: [0] * len(self.queries) for metric in self.metrics.keys()}

        for eval_idx in range(self.eval_number):
            dict_scores = self.comparator.run_evaluation_pipeline(self.queries,
                                                                  self.ground_truth,
                                                                  rag_answers,
                                                                  rag_name=rag_name)
            for metric in self.metrics.keys():
                try:
                    final_scores[metric] = (dict_scores[metric]["mean"] + final_scores[metric] * eval_idx) / (eval_idx + 1)
                    for i in range(len(dict_scores[metric]["total_evaluations"])):
                        final_evaluation[metric][i] = (dict_scores[metric]["total_evaluations"][i] * eval_idx) / (eval_idx + 1)
                except Exception as e:
                    final_scores[metric] = None

        return final_scores, final_evaluation
    

    def run_evaluations(self, log_file):

        with open(log_file, "r") as f:
            data_logs = json.load(f)

        num_rags = len(self.rag_list)
        all_scores = {metric: np.zeros(num_rags) for metric in self.metrics}
        all_scores_dict = {
            rag: {metric: 0 for metric in self.metrics} for rag in self.rag_list
        }
        all_evaluations = {rag: {metric: [0] * len(self.queries) for metric in self.metrics} for rag in self.rag_list}
        progress_bar = ProgressBar(total=num_rags, desc="Ground Truth Comparison")
        n = len(self.rag_list)
        for i, rag in enumerate(self.rag_list):
            progress_bar.update(
                i - 1,
                text=f"Processing ground truth comparisons for {rag} rag ({i+1} / {n})",
            )
            rag_answers = [row["ANSWER"] for row in self.dataframe[rag]]
            scores, evaluations = self.process_evaluation(rag_answers, rag_name=rag)
            for metric in self.metrics:
                all_scores[metric][i] = scores[metric]
                all_scores_dict[rag][metric] = scores[metric]
                all_evaluations[rag][metric] = evaluations[metric]

            data_logs["Ground Truth comparison"] = int(((i + 1) / n) * 100)
            with open(log_file, "w") as f:
                json.dump(data_logs, f)

        progress_bar.success("Ground truth comparison done")

        self.all_scores_dict = all_scores_dict

        return all_scores, all_evaluations


class ContextRelevanceComparator:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        agent: Agent,
        model: str,
        eval_number=1,
        max_attempts=5,
        batch_size=10,
    ):

        self.agent = agent
        self.model = model
        self.evaluator = ContextRelevanceEvaluator(
            agent=self.agent, model=model,
            max_attemps=max_attempts, batch_size=batch_size
        )
        self.eval_number = eval_number

        self.dataframe = dataframe
        self.queries = dataframe["QUERIES"]
        self.rag_list = [
            col
            for col in dataframe.columns
            if col != "QUERIES" and col != "GROUND_TRUTH"
        ]

    def process_evaluation(self, contexts: list[list[Chunk]]):
        final_scores = 0
        mean = 0

        model_contexts = [concat_chunks(context) for context in contexts]
        for eval_idx in range(self.eval_number):

            result = self.evaluator.run_evaluation_pipeline(
                self.queries, model_contexts
            )

            try:
                final_scores = (result + mean * eval_idx) / (eval_idx + 1)
            except Exception:
                continue

        return final_scores

    def run_evaluations(self, log_file):
        with open(log_file, "r") as f:
            data_logs = json.load(f)

        num_rags = len(self.rag_list)
        all_scores = {f"{rag}": 0.0 for rag in self.rag_list}

        progress_bar = ProgressBar(
            total=num_rags, desc="Context Relevance comparison progress"
        )
        n = len(self.rag_list)
        for i, rag in enumerate(self.rag_list):
            progress_bar.update(
                i - 1,
                text=f"Processing context relevance evaluations for {rag} rag ({i+1} / {n})",
            )
            rag_contexts = [row["CONTEXT"] for row in self.dataframe[rag]]
            score = self.process_evaluation(rag_contexts)

            all_scores[rag] = score

            data_logs["context relevance"] = int(((i + 1) / n) * 100)
            with open(log_file, "w") as f:
                json.dump(data_logs, f)

        progress_bar.success("Context relevance evaluations done")
        return all_scores


class ContextFaithfulnessComparator:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        agent: Agent,
        model: str,
        eval_number=1,
        max_attempts=5,
        batch_size=10,
    ):

        self.agent = agent
        self.model = model
        self.evaluator = ContextFaithfulnessEvaluator(
            agent=self.agent, model = model, max_attempts=max_attempts, batch_size=batch_size
        )
        self.eval_number = eval_number

        self.dataframe = dataframe
        self.queries = dataframe["QUERIES"].tolist()

        self.rag_list = [
            col
            for col in dataframe.columns
            if col != "QUERIES" and col != "GROUND_TRUTH"
        ]
        self.ground_truth = dataframe["GROUND_TRUTH"].tolist()

    def process_evaluation(self, answers: list[str], contexts: list[list[Chunk]]):
        final_scores = 0
        mean = 0

        model_contexts = [concat_chunks(context) for context in contexts]
        for eval_idx in range(self.eval_number):

            result = self.evaluator.run_evaluation_pipeline(
                self.queries, answers, model_contexts
            )

            try:
                final_scores = (result + mean * eval_idx) / (eval_idx + 1)
            except Exception:
                continue

        return final_scores

    def run_evaluations(self, log_file):
        with open(log_file, "r") as f:
            data_logs = json.load(f)

        num_rags = len(self.rag_list)
        all_scores = {f"{rag}": 0.0 for rag in self.rag_list}

        progress_bar = ProgressBar(
            total=num_rags, desc="Context faithfulness comparison progress"
        )
        n = len(self.rag_list)
        for i, rag in enumerate(self.rag_list):
            progress_bar.update(
                i - 1,
                text=f"Processing context faithfulness evaluations for {rag} rag ({i+1} / {n})",
            )
            contexts = [row["CONTEXT"] for row in self.dataframe[rag]]
            score = self.process_evaluation(self.ground_truth, contexts)
            all_scores[rag] = score

            data_logs["Context faithfulness"] = int(((i + 1) / n) * 100)
            with open(log_file, "w") as f:
                json.dump(data_logs, f)

        progress_bar.success("Context faithfulness evaluations done")
        return all_scores
    

class nDCGComparator:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        agent: Agent,
        model: str,
        eval_number=1,
        max_attempts=5,
        batch_size=10,
    ):

        self.agent = agent
        self.model = model
        self.evaluator = nDCGEvaluator(
            agent=self.agent, model=model, 
            max_attempts=max_attempts, batch_size=batch_size
        )
        self.eval_number = eval_number

        self.dataframe = dataframe
        self.queries = dataframe["QUERIES"].tolist()

        self.rag_list = [
            col
            for col in dataframe.columns
            if col != "QUERIES" and col != "GROUND_TRUTH"
        ]
        self.ground_truth = dataframe["GROUND_TRUTH"].tolist()

    def process_evaluation(self, answers: list[str], contexts: list[list[dict]]):
        final_scores = 0
        mean = 0

        for eval_idx in range(self.eval_number):

            result = self.evaluator.run_evaluation_pipeline(
                self.queries, answers, contexts
            )

            try:
                final_scores = (result + mean * eval_idx) / (eval_idx + 1)
            except Exception:
                continue

        return final_scores

    def run_evaluations(self, log_file):
        with open(log_file, "r") as f:
            data_logs = json.load(f)

        num_rags = len(self.rag_list)
        all_scores = {f"{rag}": 0.0 for rag in self.rag_list}

        progress_bar = ProgressBar(
            total=num_rags, desc="nDCG comparison progress"
        )
        n = len(self.rag_list)
        for i, rag in enumerate(self.rag_list):
            progress_bar.update(
                i - 1,
                text=f"Processing nDCG evaluations for {rag} rag ({i+1} / {n})",
            )
            contexts = [row["CONTEXT"] for row in self.dataframe[rag]]
            
            if all(context == [] for context in contexts): #to deal with the empty contexts of the naive chatbot 
                score=0
            else:
                score = self.process_evaluation(self.ground_truth, contexts)
            all_scores[rag] = score*100

            data_logs["nDCG score"] = int(((i+1)/n)*100)
            with open(log_file, "w") as f:
                json.dump(data_logs, f)

        progress_bar.success("nDCG evaluations done")
        return all_scores



