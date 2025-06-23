from .base_classes import Evaluator, MetricAnswer, GroundTruthAnswer
from .utils import process_prompt_to_json
import numpy as np
import re
from ..utils.agent import Agent
from .prompts import PROMPTS



class GroundTruthComparison(Evaluator):
    def __init__(
        self,
        agent: Agent,
        max_attemps: int = 5,
        batch_size: int = 10,
    ) -> None:
        """
        This class enables to rate RAG agents thanks to LLM as a judge method from given question where answers are already known.
        It generates a note between 0 and 5 for each rag answers and compute the mean over all queries.
        model : LLM used to judge the RAG
        """
        super().__init__(agent, max_attemps, batch_size)
        self.agent = agent

        self.system_prompt = PROMPTS[self.language]["rate_from_ground_truth"][
            "SYSTEM_PROMPT"
        ]
        self.prompt_template = PROMPTS[self.language]["rate_from_ground_truth"][
            "QUERY_TEMPLATE"
        ]
        self.metrics: dict = PROMPTS[self.language]["gt_metrics"]

    def _get_prompts(
        self,
        metric: str,
        queries: list[str],
        real_answers: list[str],
        model_answers: list[str],
    ) -> tuple[list[str], str]:
        system_prompt = self.system_prompt
        prompts = [
            self.prompt_template.replace("{real_answer}", str(real_answer))
            .replace("{query}", str(query))
            .replace("{model_answer}", str(model_answer))
            .replace("{metric}", str(metric))
            for query, real_answer, model_answer in zip(
                queries, real_answers, model_answers
            )
        ]

        return prompts, system_prompt

    def _get_evaluations_for_specific_metric(
        self,
        metric: str,
        queries: list[str],
        real_answers: list[str],
        proposed_answers: list[str],
    ) -> list[GroundTruthAnswer | None]:
        prompts, system_prompt = self._get_prompts(
            metric, queries, real_answers, proposed_answers
        )

        cleaned_outputs = [
            process_prompt_to_json(
                prompt, system_prompt, self.max_attempts, self.agent, GroundTruthAnswer
            )
            for prompt in prompts
        ]
        return cleaned_outputs

    def _clean_output(self, output: str) -> str | None:
        accepted_answers = re.compile(r"\b([0-5])\b")

        match = accepted_answers.search(output.strip())
        cleaned_outputs = float(match.group(1)) if match else None

        return cleaned_outputs

    def run_evaluation_pipeline(
        self,
        queries: list[str],
        real_answers: list[str],
        model_answers: list[str],
    ) -> dict[str, float | None]:

        all_scores = {}

        for metric, description in self.metrics.items():
            try:
                metric_description = f"{metric} : {description}"
                evaluations = self._get_evaluations_for_specific_metric(
                    metric_description, queries, real_answers, model_answers
                )

                if evaluations is None or len(evaluations) == 0:
                    all_scores[metric] = None
                    continue

                total_evaluations = [
                    float(evaluation.score)
                    for evaluation in evaluations
                    if evaluation is not None
                ]
                if len(total_evaluations) == 0:
                    all_scores[metric] = None
                else:
                    score_metric = np.mean(total_evaluations)
                    all_scores[metric] = score_metric

            except Exception as e:
                print(
                    f"Error while processing the evaluations for the metric {metric.split(':')[0]} :",
                    e,
                )
                all_scores[metric] = None

        return all_scores


class MetricComparaison(Evaluator):
    def __init__(
        self,
        agent: Agent,
        max_attemps=5,
        batch_size=10,
    ):
        super().__init__(agent, max_attemps, batch_size)

        self.agent = agent

        self.system_prompt = PROMPTS[self.language]["rate_metric"]["SYSTEM_PROMPT"]
        self.prompt_template = PROMPTS[self.language]["rate_metric"]["QUERY_TEMPLATE"]
        self.metrics = PROMPTS[self.language]["metrics"]

    def _get_prompts(
        self,
        metric: str,
        queries: list[str],
        ground_truths: list[str],
        first_answers: list[str],
        second_answers: list[str],
    ) -> tuple[list[str], str]:

        system_prompt = self.system_prompt.replace("{metric}", str(metric))
        system_prompt = self.system_prompt.replace("{metric}", str(metric))
        prompts = [
            self.prompt_template.replace("{query}", str(query))
            .replace("{ground_truth}", str(ground_truth))
            .replace("{answer_a}", str(first_answer))
            .replace("{answer_b}", str(second_answer))
            for query, first_answer, ground_truth, second_answer in zip(
                queries, first_answers, ground_truths, second_answers
            )
        ]
        return prompts, system_prompt

    def _clean_output(self, output):
        accepted_answers = re.compile(r"\b(A|B)\b", re.IGNORECASE)

        match = accepted_answers.search(output)
        cleaned_output = str(match.group(1)) if match else None

        return cleaned_output

    def _get_evaluations_for_specific_metric(
        self,
        metric: str,
        queries: list[str],
        ground_truths: list[str],
        first_answers: list[str],
        second_answers: list[str],
    ) -> list[MetricAnswer | None]:
        prompts, system_prompt = self._get_prompts(
            metric, queries, ground_truths, first_answers, second_answers
        )

        cleaned_outputs = [
            process_prompt_to_json(
                prompt, system_prompt, self.max_attempts, self.agent, MetricAnswer
            )
            for prompt in prompts
        ]
        return cleaned_outputs

    def run_evaluation_pipeline(
        self,
        queries: list[str],
        ground_truths: list[str],
        first_answers: list[str],
        second_answers: list[str],
    ) -> dict[str, tuple[float | None, float | None]]:
        all_scores = {}

        for metric, description in self.metrics.items():
            try:
                metric_description = f"{metric} : {description}"
                evaluations = self._get_evaluations_for_specific_metric(
                    metric_description,
                    queries,
                    ground_truths,
                    first_answers,
                    second_answers,
                )

                if evaluations is None or len(evaluations) == 0:
                    all_scores[metric] = (None, None)
                    continue

                total_evaluations = (
                    sum(1 for eval in evaluations if eval is not None) or 1
                )
                score_first_rag = (
                    sum(1 for eval in evaluations if eval.winner == "A")
                    / total_evaluations
                ) * 100
                score_second_rag = (
                    sum(1 for eval in evaluations if eval.winner == "B")
                    / total_evaluations
                ) * 100

                all_scores[metric] = (score_first_rag, score_second_rag)
            except Exception as e:
                print(
                    f"Error while processing the evaluations for the metric {metric} :",
                    e,
                )
                all_scores[metric] = (None, None)
        return all_scores
