from openai import OpenAI
from mistralai import Mistral
import requests
import numpy as np
import json
import cv2
import io
from pydantic import BaseModel
from typing import Union
from ecologits import EcoLogits
from .progress import ProgressBar


def np_array_to_file(image_np: np.ndarray, format: str = "jpeg"):
    success, encoded_image = cv2.imencode(f".{format}", image_np)
    if not success:
        raise ValueError("Failed to encode image.")
    image_bytes = io.BytesIO(encoded_image.tobytes())
    image_bytes.name = f"image.{format}"  # Nécessaire pour requests
    return image_bytes


tuple_delimiter = ","
start_delimiter = "##"
end_delimiter = "##"
eval_pos = "OUI"
eval_neg = "NON"
list_entities = (
    "[rôle, humain, évènement, journal, entreprise, date, lieu, objet, chiffre]"
)


def predict_json(
    system_prompt: str,
    prompt: str,
    model: str,
    client: Union[OpenAI, Mistral],
    json_format: BaseModel,
    temperature: float = None,
) -> str:
    try:
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "response_format": json_format,
        }
        # Only add temperature if not None
        if temperature is not None:
            params["temperature"] = temperature
        if type(client) is OpenAI:
            response = client.beta.chat.completions.parse(**params)

        if type(client) is Mistral:
            response = client.chat.parse(**params)

        json_response = response.choices[0].message
        if json_response.parsed:

            return json_response.parsed
        else:
            print("refusal ", json_response.refusal)

    except Exception as e:
        print(f"Error: {e}")


def predict(
    system_prompt: str, prompt: str, model: str, client: OpenAI, temperature: float = 0
) -> str:
    """
    Gives the prompt to the LLM and returns the output
    """
    EcoLogits.init()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    try:
        answer = response.choices[0].message.content

    except Exception:
        print("Error in answer generation but still running")
        answer = ""

    try:
        input_tokens, output_tokens = (
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    except Exception:
        input_tokens, output_tokens = 0, 0
        print("Error with token count but still running")

    try:
        impacts = [
            response.impacts.gwp.value.min,
            response.impacts.gwp.value.max,
            response.impacts.gwp.unit,
        ]
    except Exception as e:
        impacts = [0, 0, ""]

    # Bug return Exception when using with Ollama
    try:
        energy = [
            response.impacts.energy.value.min,
            response.impacts.energy.value.max,
            response.impacts.energy.unit,
        ]
    except Exception as e:
        energy = [0, 0, ""]
    return {
        "texts": answer,
        "nb_input_tokens": input_tokens,
        "nb_output_tokens": output_tokens,
        "impacts": impacts,
        "energy": energy,
    }


def predict_vllm(
    system_prompts: list[str],
    prompts: list[str],
    model: str,
    url: str,
    temperature: float = 0,
    images: list[str] = None,
    json_format=None,
) -> tuple[list[str], int, int]:
    """
    Gives the prompts to the LLM and returns the outputs
    """

    if type(prompts) is type("str"):
        prompts = [prompts]

    if type(system_prompts) is type("str"):
        system_prompts = [system_prompts]

    if json_format is not None:
        json_format = json.dumps(json_format.model_json_schema())

    data = {
        "model_name": model,
        "systems": system_prompts,
        "prompts": prompts,
        "temperature": temperature,
        "json_format": json_format,
    }
    if images is None or images[0] is None:
        answers = requests.post(url, data=data).json()
    else:
        files = []
        for i in range(len(images)):
            for j in range(len(images[i])):
                files.append(
                    (
                        "images",
                        (str(i), np_array_to_file(image_np=images[i][j]), "image/jpeg"),
                    )
                )
        answers = requests.post(url, data=data, files=files).json()
    return answers


def multiple_predict(
    system_prompts: list[str],
    prompts: list[str],
    model: str,
    client: OpenAI,
    temperature: float = 0,
) -> tuple[list[str], int, int]:
    """
    Gives the prompts to the LLM and returns the outputs
    """
    answers, input_tokens, output_tokens, impacts, energy = (
        [],
        0,
        0,
        [0, 0, ""],
        [0, 0, ""],
    )

    if type(prompts) is type("str"):
        prompts = [prompts]

    if type(system_prompts) is type("str"):
        system_prompts = [system_prompts]

    progress_bar = ProgressBar(prompts)
    for k, (prompt, system_prompt) in enumerate(zip(prompts, system_prompts)):
        progress_bar.update(
            index=k, text=f"Processing prompt {k+1}/{progress_bar.total}"
        )
        answer = predict(system_prompt, prompt, model, client, temperature)
        answers.append(answer["texts"])
        input_tokens += answer["nb_input_tokens"]
        output_tokens += answer["nb_output_tokens"]
        impacts = answer["impacts"]
        energy = answer["energy"]
    progress_bar.clear()
    return {
        "texts": answers,
        "nb_input_tokens": input_tokens,
        "nb_output_tokens": output_tokens,
        "impacts": impacts,
        "energy": energy,
    }


def predict_mistral(
    system_prompt: str, prompt: str, model: str, client: Mistral, temperature: float = 0
) -> str:
    """
    Gives the prompt to the LLM and returns the output
    """
    EcoLogits.init()
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    try:
        answer = response.choices[0].message.content

    except Exception:
        print("Error in answer generation but still running")
        answer = ""

    try:
        input_tokens, output_tokens = (
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    except Exception:
        input_tokens, output_tokens = 0, 0
        print("Error with token count but still running")

    try:

        impacts = [
            response.impacts.gwp.value.min,
            response.impacts.gwp.value.max,
            response.impacts.gwp.unit,
        ]
    except Exception:
        impacts = [0, 0, ""]

    try:
        energy = [
            response.impacts.energy.value.min,
            response.impacts.energy.value.max,
            response.impacts.energy.unit,
        ]
    except Exception as e:
        energy = [0, 0, ""]
    return {
        "texts": answer,
        "nb_input_tokens": input_tokens,
        "nb_output_tokens": output_tokens,
        "impacts": impacts,
        "energy": energy,
    }


def multiple_predict_mistral(
    system_prompts: list[str],
    prompts: list[str],
    model: str,
    client: OpenAI,
    temperature: float = 0,
) -> tuple[list[str], int, int]:
    """
    Gives the prompts to the LLM and returns the outputs
    """
    answers, input_tokens, output_tokens, impacts, energy = (
        [],
        0,
        0,
        [0, 0, ""],
        [0, 0, ""],
    )

    if type(prompts) is type("str"):
        prompts = [prompts]

    if type(system_prompts) is type("str"):
        system_prompts = [system_prompts]

    progress_bar = ProgressBar(prompts)
    for k, (prompt, system_prompt) in enumerate(zip(prompts, system_prompts)):
        answer = predict_mistral(system_prompt, prompt, model, client, temperature)
        progress_bar.update(
            index=k, text=f"Processing prompt {k+1}/{progress_bar.total}"
        )
        answers.append(answer["texts"])
        input_tokens += answer["nb_input_tokens"]
        output_tokens += answer["nb_output_tokens"]
        impacts = answer["impacts"]
        energy = answer["energy"]
    progress_bar.clear()

    return {
        "texts": answers,
        "nb_input_tokens": input_tokens,
        "nb_output_tokens": output_tokens,
        "impacts": impacts,
        "energy": energy,
    }


def get_system_prompt(config_server: dict, prompts: dict) -> str:

    system_prompt = ""
    system_prompt_name = config_server["local_params"]["generation_system_prompt_name"]

    if config_server["local_params"]["forced_system_prompt"]:
        system_prompt = config_server["all_system_prompt"][system_prompt_name]
    else:
        if config_server["local_params"]["generation_system_prompt_name"] != "default":
            system_prompt = config_server["all_system_prompt"][system_prompt_name]
        else:
            system_prompt = prompts["smooth_generation"]["SYSTEM_PROMPT"]

    return system_prompt
