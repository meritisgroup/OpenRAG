from .base_classes import Evaluator, StatementSupported
from .utils import process_prompt, process_prompt_to_json
import numpy as np
import re
from ..utils.agent import Agent
from .prompts import PROMPTS
from backend.database.rag_classes import Chunk


class ContextRelevanceEvaluator(Evaluator):
    def __init__(
        self,
        agent: Agent,
        max_attemps: int = 5,
        batch_size: int = 10,
    ) -> None:
        """
        This class enables to rate RAG agents thanks to LLM as a judge method from given question where answers are already known.
        model : LLM used to judge the RAG
        """
        super().__init__(agent, max_attemps, batch_size)
        self.agent = agent

        self.system_prompt: str = PROMPTS[self.language]["rate_context_relevance"][
            "SYSTEM_PROMPT"
        ]
        self.prompt_template: str = PROMPTS[self.language]["rate_context_relevance"][
            "QUERY_TEMPLATE"
        ]

    def _get_prompts(
        self,
        queries: list[str],
        model_contexts: list[str],
    ) -> tuple[list[str], str]:
        system_prompt = self.system_prompt
        prompts = [
            self.prompt_template.replace("{query}", str(query)).replace(
                "{model_context}", str(model_context)
            )
            for query, model_context in zip(queries, model_contexts)
        ]

        return prompts, system_prompt

    def _clean_output(self, output: str) -> str:
        insufficient_answer = re.compile(r"Insufficient Information", re.IGNORECASE)
        empty_output = ""
        if insufficient_answer.search(output):
            return empty_output
        elif re.compile(r"Informations insuffisantes", re.IGNORECASE).search(output):
            return empty_output
        else:
            return output.strip()

    def _get_evaluation(
        self,
        queries: list[str],
        model_contexts: list[str],
    ) -> list[float | None]:
        prompts, system_prompt = self._get_prompts(queries, model_contexts)

        cleaned_outputs = [
            process_prompt(
                prompt,
                system_prompt,
                self.max_attempts,
                self.agent,
                self._clean_output,
            )
            for prompt in prompts
        ]
        context_relevance_evaluations = []
        for clean_output, model_context in zip(cleaned_outputs, model_contexts):
            if len(clean_output) > 0:
                context_relevance_evaluations.append(
                    len(clean_output.split(".")) * 100 / len(model_context.split("."))
                )
            else:
                context_relevance_evaluations.append(0)

        return context_relevance_evaluations

    def run_evaluation_pipeline(
        self, queries: list[str], model_contexts: list[str]
    ) -> float:
        context_relevance_evaluations = self._get_evaluation(queries, model_contexts)

        final_mean_context_relevance_evaluation = np.mean(context_relevance_evaluations)

        return final_mean_context_relevance_evaluation


class ContextFaithfulnessEvaluator(Evaluator):
    def __init__(
        self,
        agent: Agent,
        max_attemps: int = 5,
        batch_size: int = 10,
    ) -> None:
        """
        This class enables to rate RAG agents thanks to LLM as a judge method from given question where answers are already known.
        model : LLM used to judge the RAG
        """
        super().__init__(agent, max_attemps, batch_size)
        self.agent = agent
        self.max_attempts = max_attemps

        # Templates to retrieve statements from queries and their answers
        self.system_prompt_statements: str = PROMPTS[self.language]["get_statements"][
            "SYSTEM_PROMPT"
        ]
        self.prompt_template_statements: str = PROMPTS[self.language]["get_statements"][
            "QUERY_TEMPLATE"
        ]
        self.output_delimiter: str = PROMPTS[self.language]["output_delimiter"]
        self.expected_results: str = PROMPTS[self.language]["expected_result"]
        self.good_result: str = PROMPTS[self.language]["good_result"]
        # Templates to evaluate every statements retrieved
        self.system_prompt_faithfulness: str = PROMPTS[self.language][
            "rate_context_faithfulness"
        ]["SYSTEM_PROMPT"]
        self.prompt_template_faithfulness: str = PROMPTS[self.language][
            "rate_context_faithfulness"
        ]["QUERY_TEMPLATE"]

    def format_statements(self, statement_list: list[str]) -> list[str]:
        formated_statement_list = ""

        for i, statement in range(statement_list):
            formated_statement_list += f"statement {i}: {statement} \n"

        return formated_statement_list

    def _get_prompts_statements(
        self, queries: list[str], answers: list[str]
    ) -> tuple[list[str], str]:
        system_prompt = self.system_prompt_statements
        prompts = [
            self.prompt_template_statements.replace("{query}", str(query)).replace(
                "{answer}", str(answer)
            )
            for query, answer in zip(queries, answers)
        ]

        return prompts, system_prompt

    def _clean_output_statements(self, output: str) -> list[str]:
        statements = output.split(self.output_delimiter)

        if len(statements) > 0:
            return statements[1:]
        else:
            return None

    def _get_statements(self, queries: list[str], answers: list[str]):
        prompts, system_prompt = self._get_prompts_statements(queries, answers)
        statements: list[str] = [
            process_prompt(
                prompt,
                system_prompt,
                self.max_attempts,
                self.agent,
                self._clean_output_statements,
            )
            for prompt in prompts
        ]

        return statements

    def _get_prompts_faithfulness(
        self,
        statements: list[str],
        model_context: str,
    ) -> tuple[list[str], str]:
        system_prompt = self.system_prompt_faithfulness

        prompts = [
            self.prompt_template_faithfulness.replace(
                "{model_context}", str(model_context)
            ).replace("{statement}", str(statement))
            for statement in statements
        ]

        return prompts, system_prompt

    def _clean_output_faithfulness(self, output: str) -> str | None:
        accepted_answer = re.compile(self.expected_results, re.IGNORECASE)
        match = accepted_answer.search(output)
        return match.group(0) if match else None

    def _get_evaluation_faithfulness(
        self,
        statements: list[str],
        model_context: str,
    ) -> list[float | None]:
        prompts, system_prompt = self._get_prompts_faithfulness(
            statements, model_context
        )
        cleaned_outputs: list[bool] = [
            process_prompt_to_json(
                prompt, system_prompt, self.max_attempts, self.agent, StatementSupported
            )
            for prompt in prompts
        ]

        return cleaned_outputs

    def run_evaluation_pipeline(
        self, queries: list[str], answers: list[str], model_contexts: list[str]
    ) -> int | None:

        statements_list = self._get_statements(queries, answers)

        mean_score = 0
        valid_queries = 0

        for i, statements in enumerate(statements_list):
            if model_contexts[i] == "":
                return None
            else:
                results: list[StatementSupported] = self._get_evaluation_faithfulness(
                    statements, model_contexts[i]
                )
                try:
                    if any(results):
                        query_score = [result.supported for result in results].count(
                            True
                        ) / len(results)
                        valid_queries += 1
                        mean_score = (
                            query_score + mean_score * (valid_queries - 1)
                        ) / valid_queries
                    else:
                        return None
                except Exception as e:
                    print(
                        f"Encountered a problem while evaluating the faithfulness of the context of the query {queries[i]}"
                    )
                    return None

        return int(mean_score * 100) if valid_queries > 0 else None
