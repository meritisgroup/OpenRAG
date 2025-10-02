from ..utils.agent import Agent
from pydantic import BaseModel
from typing import List


def process_prompts(
    prompts: List[str],
    system_prompts: List[str],
    max_retry: int,
    agent,
    clean_output,
):
    """
    Traite une liste de prompts avec un agent et une fonction de nettoyage.
    Réessaie les prompts échoués jusqu'à max_retry fois.
    """

    if max_retry <= 0 or not prompts:
        return None

    if len(system_prompts) == 1 and len(prompts) > 1:
        system_prompts = [system_prompts[0] for _ in range(len(prompts))]

    evaluations = agent.multiple_predict(system_prompts, prompts)["texts"]
    if evaluations is None:
        return None

    cleaned_outputs = [clean_output(ev) if ev is not None else None
                       for ev in evaluations]

    failed_indices = [i for i, result in enumerate(cleaned_outputs) if result is None]

    if not failed_indices:
        return cleaned_outputs

    failed_prompts = [prompts[i] for i in failed_indices]
    failed_system_prompts = [system_prompts[i] for i in failed_indices]

    retry_eval = process_prompts(failed_prompts,
                                 failed_system_prompts,
                                 max_retry - 1,
                                 agent,
                                 clean_output)

    if retry_eval is not None:
        merged = []
        i_retry = 0
        for result in cleaned_outputs:
            if result is None:
                merged.append(retry_eval[i_retry])
                i_retry+=1
            else:
                merged.append(result)
        return merged

    return cleaned_outputs


def process_prompts_to_json(
    prompts: List[str],
    system_prompts: List[str],
    max_retry: int,
    agent,
    json_format: BaseModel,
) -> BaseModel | None:
    if max_retry <= 0 or not prompts:
        return None
    if len(system_prompts)==1 and len(prompts)>1:
        system_prompts = [system_prompts[0] for i in range(len(prompts))]
        
    evaluation = agent.multiple_predict_json(system_prompts,
                                             prompts,
                                             json_format)

    if evaluation is None:
        return None

    failed_prompts = [
        prompt for prompt, result in zip(prompts, evaluation)
        if result is None
    ]

    if not failed_prompts:
        return evaluation

    retry_eval = process_prompts_to_json(
        failed_prompts,
        system_prompts,
        max_retry - 1,
        agent,
        json_format,
    )

    if retry_eval is not None:
        merged = []
        i_retry = 0
        for result in evaluation:
            if result is None:
                merged.append(retry_eval[i_retry])
                i_retry += 1
            else:
                merged.append(result)
        return merged
    return evaluation

