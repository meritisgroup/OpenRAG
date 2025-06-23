from ..utils.agent import Agent
from pydantic import BaseModel


def process_prompt(
    prompt: str,
    system_prompt: str,
    max_retry: str,
    agent: Agent,
    clean_output,
):

    for _ in range(max_retry):
        evaluation = agent.predict(prompt, system_prompt)
        cleaned_output = clean_output(evaluation["texts"])
        if cleaned_output is not None:
            return cleaned_output

    return None


def process_prompt_to_json(
    prompt: str,
    system_prompt: str,
    max_retry: str,
    agent: Agent,
    json_format: BaseModel,
) -> BaseModel | None:
    for _ in range(max_retry):
        evaluation = agent.predict_json(system_prompt, prompt, json_format)
        if evaluation is not None:
            return evaluation

    return None
