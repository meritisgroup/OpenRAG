from vllm import SamplingParams
import math
import base64
import cv2
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoProcessor


def np_array_to_base64_data_uri(image_np: np.ndarray, format: str = "jpeg") -> str:
    success, encoded_image = cv2.imencode(f".{format}", image_np)
    if not success:
        raise ValueError("Image encoding failed.")

    base64_str = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{base64_str}"


def get_sampling_params(model_name, max_tokens=512, temperature=0, tokenizer=None, json_format=None):
    if json_format is not None:
        guided_decoding_params_json = GuidedDecodingParams(json=json_format)
    else:
        guided_decoding_params_json = None
        

    if model_name=="microsoft/phi-4":
        return SamplingParams(temperature=temperature,
                              max_tokens=max_tokens,
                              min_p=0.15,
                              top_p=0.85,
                              stop="<|end|>",
                              logprobs=3,
                              guided_decoding=guided_decoding_params_json)
    elif model_name=="google/gemma-2-2b-it" or model_name=="google/gemma-2-9b-it" or model_name=="google/gemma-2-27b-it":
        return SamplingParams(temperature=temperature,
                              max_tokens=max_tokens,
                              min_p=0.15,
                              top_p=0.85,
                              stop="<end_of_turn>",
                              logprobs=3,
                              guided_decoding=guided_decoding_params_json)
    elif model_name=="meta-llama/Llama-3.1-8B-Instruct" or model_name=="google/gemma-3-1b-it" or model_name=="google/gemma-3-4b-it" or model_name=="google/gemma-3-12b-it":
        return SamplingParams(temperature=temperature,
                              max_tokens=max_tokens,
                              min_p=0.15,
                              top_p=0.85,
                              logprobs=3)
    elif model_name=="Qwen/Qwen2.5-VL-3B-Instruct" or model_name=="Qwen/Qwen2.5-VL-7B-Instruct" or model_name=="Qwen/Qwen2.5-VL-32B-Instruct" or model_name=="Qwen/Qwen2.5-VL-72B-Instruct":
        return SamplingParams(temperature=temperature,
                              min_p=0.15,
                              top_p=0.85,
                              max_tokens=max_tokens,
                              stop_token_ids=[],
                              logprobs=3)
    elif model_name=="openbmb/MiniCPM-V-2_6":
            stop_tokens = ['<|im_end|>', '<|endoftext|>']
            stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
            return SamplingParams(temperature=0.2,
                                max_tokens=1024,
                                stop_token_ids=stop_token_ids)
    
def get_chat_template(model_name, tokenizer,
                      prompts, systems, images=[]):
    prompts_results = []
    for prompt, system, image in zip(prompts, systems, images):
        if model_name == "microsoft/phi-4":
            temp = """<|im start|>system<|im sep|>{}<|im end|><|im start|>user<|im sep|>{}<|im end|><|im start|>assistant<|im sep|>""".format(
                system, prompt
            )
            prompts_results.append(temp)
        elif model_name == "meta-llama/Llama-3.1-8B-Instruct":
            temp = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            temp = tokenizer.apply_chat_template(
                temp, tokenize=False, add_generation_prompt=True
            )
            prompts_results.append(temp)
        elif model_name=="openbmb/MiniCPM-V-2_6":
            temp = []
            for i in range(len(image)):
                temp.append(Image.fromarray(image[i]))
            temp.append(prompt)
            inputs = [{'role': 'user', 'content': temp}]
            prompts_results.append(inputs)


        elif model_name=="google/gemma-3-1b-it" or model_name=="google/gemma-3-4b-it" or model_name=="google/gemma-3-12b-it":
            temp = [{"role": "system", "content": [{"type": "text", "text":system}]},
                    {"role": "user", "content": [{"type": "text", "text":prompt}]}]
            temp = tokenizer.apply_chat_template(temp,
                                                 tokenize=False,
                                                 add_generation_prompt=True)
            prompts_results.append(temp)
        elif (
            model_name == "google/gemma-2-2b-it"
            or model_name == "google/gemma-2-9b-it"
            or model_name == "google/gemma-2-27b-it"
        ):
            prompt = "Instruction:{}\n\n Prompt:{}".format(system, prompt)
            temp = [{"role": "user", "content": prompt}]
            temp = tokenizer.apply_chat_template(
                temp, tokenize=False, add_generation_prompt=True
            )
            prompts_results.append(temp)
        elif (
            model_name == "Qwen/Qwen2.5-VL-3B-Instruct"
            or model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
            or model_name == "Qwen/Qwen2.5-VL-32B-Instruct"
            or model_name == "Qwen/Qwen2.5-VL-72B-Instruct"
        ):
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            if len(image) > 0:
                for i in range(len(image)):
                    messages[1]["content"].append(
                        {
                            "type": "image",
                            "image": np_array_to_base64_data_uri(image_np=image[i]),
                            "min_pixels": 224 * 224,
                            "max_pixels": 1280 * 28 * 28,
                        }
                    )
            processor = AutoProcessor.from_pretrained(model_name)
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }
            prompts_results.append(llm_inputs)

    return prompts_results


def replace_inf(data, replacement=None):
    if isinstance(data, float) and math.isinf(data):
        return replacement
    elif isinstance(data, list):
        return [replace_inf(item, replacement) for item in data]
    elif isinstance(data, dict):
        return {key: replace_inf(value, replacement) for key, value in data.items()}
    return data


