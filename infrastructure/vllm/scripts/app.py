from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from typing import Optional, List
from typing import List
import numpy as np
import gc
import json
from vllm import LLM, EngineArgs
import torch
from dataclasses import asdict
from utils import get_sampling_params, get_chat_template, replace_inf
from base_classe import LoadModelBase, EmbeddingsBase, ReleaseBase
from base_classe import RerankingBase, PredictBase, HFKey
from contextlib import asynccontextmanager
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers import ColPaliForRetrieval, ColPaliProcessor
from utils import np_array_to_base64_data_uri
import os
import io
from PIL import Image
import time
import torch.nn.functional as F
import builtins
import typing
from codecarbon import EmissionsTracker

builtins.List = typing.List


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {}
    yield


app = FastAPI(lifespan=lifespan)


def loadmodel(model_name: str, gpu_memory_utilization=0.9, nb_try=0):
    try:
        if model_name not in app.state.models.keys() or "model" not in app.state.models[model_name].keys() or "tokenizer" not in app.state.models[model_name].keys():
            app.state.models[model_name] = {}
            if model_name=="meta-llama/Llama-3.1-8B-Instruct":
                app.state.models[model_name]["model"] = LLM(model=model_name,
                                                            gpu_memory_utilization=gpu_memory_utilization,
                                                            max_model_len=6000)
                app.state.models[model_name]["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
            elif model_name=="Qwen/Qwen2.5-VL-3B-Instruct" or model_name=="Qwen/Qwen2.5-VL-7B-Instruct" or model_name=="Qwen/Qwen2.5-VL-32B-Instruct" or model_name=="Qwen/Qwen2.5-VL-72B-Instruct":
                app.state.models[model_name]["model"] = LLM(model=model_name,
                                                            limit_mm_per_prompt={"image": 100,
                                                                                "video": 10},
                                                            max_model_len=58000)
                app.state.models[model_name]["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
            elif model_name=="openbmb/MiniCPM-V-2_6":
                app.state.models[model_name]["model"] = model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6',
                                                                                          trust_remote_code=True,
                                                                                          attn_implementation='sdpa', 
                                                                                          torch_dtype=torch.bfloat16) 
                app.state.models[model_name]["model"] = app.state.models[model_name]["model"].eval()
                if torch.cuda.is_available():
                    app.state.models[model_name]["model"] = app.state.models[model_name]["model"].cuda()

                app.state.models[model_name]["tokenizer"] = AutoTokenizer.from_pretrained(model_name,
                                                                                          trust_remote_code=True)
            elif model_name=="vidore/colpali-v1.2-hf":
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

                app.state.models[model_name]["model"] = ColPaliForRetrieval.from_pretrained(model_name,
                                                                                            torch_dtype=torch.bfloat16,
                                                                                            device_map=device,
                                                                                            ).eval()
                app.state.models[model_name]["processor"] = ColPaliProcessor.from_pretrained(model_name)
            elif model_name=="openbmb/VisRAG-Ret":
                app.state.models[model_name]["tokenizer"] = AutoTokenizer.from_pretrained(model_name,
                                                                                          trust_remote_code=True)
                app.state.models[model_name]["model"] = AutoModel.from_pretrained(model_name,
                                                                                  trust_remote_code=True,
                                                                                  attn_implementation="sdpa").eval().cuda()
            elif model_name=="google/gemma-3-1b-it" or model_name=="google/gemma-3-4b-it" or model_name=="google/gemma-3-12b-it":
                app.state.models[model_name]["model"] = LLM(model=model_name,
                                                            gpu_memory_utilization=gpu_memory_utilization,
                                                            max_model_len=16384)
                app.state.models[model_name]["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
            else:
                app.state.models[model_name]["model"] = LLM(model=model_name,
                                                            gpu_memory_utilization=gpu_memory_utilization)
                app.state.models[model_name]["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
                
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and nb_try==0:
            releasememorymodels()
            loadmodel(model_name=model_name,
                      nb_try=1)
        
def loadmodels(models_name: List[str], gpu_memory_utilization=0.9):
    gpu_memory_utilization = gpu_memory_utilization/len(models_name)
    for model_name in models_name:
        loadmodel(model_name=model_name, gpu_memory_utilization=gpu_memory_utilization)

def releasememorymodel(model_name: str):
    keys = list(app.state.models[model_name].keys())
    for key in keys:
        del app.state.models[model_name][key]
    del app.state.models[model_name]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

def releasememorymodels():
    keys = list(app.state.models.keys())
    for model_name in keys:
        releasememorymodel(model_name=model_name)
    time.sleep(1)
    print("release models")

def generate(
    prompts: List[str],
    systems: List[str],
    model_name: str,
    temperature: int = 0,
    images=[],
    json_format = None
):
    loadmodel(model_name=model_name)

    prompts_template = get_chat_template(model_name=model_name,
                                         tokenizer=app.state.models[model_name]["tokenizer"],
                                         prompts=prompts,
                                         systems=systems,
                                         images=images)
    sampler = get_sampling_params(model_name=model_name,
                                  temperature=temperature,
                                  tokenizer=app.state.models[model_name]["tokenizer"])
    if model_name=="openbmb/MiniCPM-V-2_6":
        generation = app.state.models[model_name]["model"].chat(
                                       image=None,
                                    msgs=prompts_template,
                                    tokenizer=app.state.models[model_name]["tokenizer"]
                            )

        nb_tokens = [0 for i in range(len(generation))]
        return generation, None, nb_tokens, nb_tokens
    else:
        generation = app.state.models[model_name]["model"].generate(prompts_template,
                                                                    sampling_params=sampler)
        
        texts = []
        nb_input_tokens = []
        nb_output_tokens = []
        for i in range(len(generation)):
            texts.append(generation[i].outputs[0].text)
            nb_input_tokens.append(len(generation[i].prompt_token_ids))
            nb_output_tokens.append(len(generation[i].outputs[0].token_ids))
        logprobs = []
        for output in generation:
            temp = []
            for i in range(len(output.outputs[0].logprobs)):
                t = {}
                for key in output.outputs[0].logprobs[i].keys():
                    t[output.outputs[0].logprobs[i][key].decoded_token] = output.outputs[0].logprobs[i][key].logprob
                temp.append(t)
            logprobs.append(temp)

        return texts, logprobs, nb_input_tokens, nb_output_tokens
        
def generate_embeddings(texts: List[str], model_name: str):
    loadmodel(model_name=model_name)

    outputs = app.state.models[model_name]["model"].encode(texts)
    nb_tokens = []
    for i in range(len(outputs)):
        nb_tokens.append(len(outputs[i].prompt_token_ids))
        outputs[i] = outputs[i].outputs.embedding
    return outputs, nb_tokens
    
def get_embedding_vlm(model_name, images=[], queries=[]):
    loadmodel(model_name=model_name)

    inputs_image = []
    inputs_text = []
    for i in range(len(queries)):
        inputs_text.append(queries[i])
        inputs_image.append(None)
    for i in range(len(images)):
        inputs_image.append(Image.fromarray(images[i]))
        inputs_text.append("Can you describe and extract the information?")
    inputs = {"text": inputs_text,
               "image": inputs_image,
              "tokenizer": app.state.models[model_name]["tokenizer"]}
    with torch.no_grad():
        outputs = app.state.models[model_name]["model"](**inputs)

    last_hidden_state = outputs.last_hidden_state
    attention_mask = outputs.attention_mask

    attention_mask_cumulative = attention_mask * attention_mask.cumsum(dim=1)
    weighted_sum = torch.sum(
        last_hidden_state * attention_mask_cumulative.unsqueeze(-1).float(), dim=1
    )
    sum_of_weights = attention_mask_cumulative.sum(dim=1, keepdim=True).float()
    weighted_mean = weighted_sum / sum_of_weights
    embeddings = F.normalize(weighted_mean, p=2, dim=1).detach().cpu().numpy().tolist()
    return embeddings, 0

def generate_embeddings_processor(model_name, images=[], queries=[]):
    loadmodel(model_name=model_name)

    embeddings = []
    with torch.no_grad():
        for i in range(len(images)):
            batch = app.state.models[model_name]["processor"](images=[Image.fromarray(images[i])]).to(app.state.models[model_name]["model"].device)
            embeddings.append(app.state.models[model_name]["model"](**batch)["embeddings"].detach().to(torch.float16).cpu().numpy().tolist())
        for i in range(len(queries)):
            batch = app.state.models[model_name]["processor"](text=[queries[i]]).to(app.state.models[model_name]["model"].device)
            embeddings.append(app.state.models[model_name]["model"](**batch)["embeddings"].detach().to(torch.float16).cpu().numpy().tolist())
    return embeddings, 0

    
def rerank(query: str, contexts: List[str], model_name: str):
    loadmodel(model_name=model_name)
    outputs = app.state.models[model_name]["model"].score(query, contexts)
    nb_inputs_token = 0
    for i in range(len(outputs)):
        nb_inputs_token + len(outputs[i].prompt_token_ids)
        outputs[i] = outputs[i].outputs.score
    return outputs, nb_inputs_token


@app.get("/")
def read_root():
    models_running = list(app.state.models.keys())
    str_running = ""
    for model_name in models_running:
        str_running += "  {}".format(model_name)
    return {"message": "models running: {}".format(str_running)}


@app.post("/load_model")
def load_model(request: LoadModelBase):
    print("Loading ", request.model_name)
    loadmodel(
        model_name=request.model_name,
        gpu_memory_utilization=request.gpu_memory_utilization,
    )


@app.post("/release_memory")
def release_memory(request: ReleaseBase):
    if request.model_name is None:
        releasememorymodels()
    else:
        releasememorymodel(model_name=request.model_name)


@app.post("/embeddings")
async def embeddings(request: EmbeddingsBase):
    loadmodel(model_name=request.model_name)
    tracker = EmissionsTracker(project_name="vllm-inference",
                               save_to_file=False)
    tracker.start()

    if not request.texts:
        raise HTTPException(
            status_code=400, detail="The texts list should not be empty."
        )
    if not request.model_name:
        raise HTTPException(status_code=400, 
                            detail="The model name should not be None.")
    
    embeddings, nb_tokens = generate_embeddings(texts=request.texts,
                                                model_name=request.model_name)
    
    co2 = tracker.stop()*1000
    impact = co2*1.65
    data = {"embeddings": embeddings,
            "nb_tokens": nb_tokens,
            "impacts": [co2, co2, " gCO2eq"],
            "energy": [impact, impact, " Wh"]}
    return data

@app.post("/embeddings_vlm")
async def embeddings_vlm(queries: List[str] = Form(None),
                         model_name: str = Form(...),
                         images: Optional[List[UploadFile]] = File(None),
                         mode: str = Form(...)):
    loadmodel(model_name=model_name)
    tracker = EmissionsTracker(project_name="vllm-inference",
                               save_to_file=False)
    tracker.start()

    if images is None:
        images = []
    if queries is None:
        queries = []

    for i in range(len(images)):
        content = await images[i].read()
        images[i] = np.array(Image.open(io.BytesIO(content)))

    if mode=="Processor":
        embeddings, nb_tokens = generate_embeddings_processor(images=images,
                                                            queries=queries,
                                                            model_name=model_name)
    else:
        embeddings, nb_tokens = get_embedding_vlm(images=images,
                                              queries=queries,
                                              model_name=model_name)
    co2 = tracker.stop()
    impact = co2*1.65
    data = {"embeddings": embeddings,
            "nb_tokens": nb_tokens,
            "impacts": [co2, co2, " gCO2eq"],
            "energy": [impact, impact, " Wh"]}
    return data


@app.post("/predict")
async def predict(
    prompts: List[str] = Form(...),
    model_name: str = Form(...),
    systems: Optional[List[str]] = Form(None),
    temperature: Optional[float] = Form(0.0),
    images: Optional[List[UploadFile]] = File(None),
    json_format: Optional[str] = Form(None)
):
    if json_format is not None:
        json_data = json.loads(json_format)

    loadmodel(model_name=model_name)
    tracker = EmissionsTracker(project_name="vllm-inference",
                               save_to_file=False)
    tracker.start()

    if not prompts:
        raise HTTPException(status_code=400, detail="Prompts list should not be empty.")
    if not model_name:
        raise HTTPException(
            status_code=400, detail="The model name should not be None."
        )
    image_contents = [[] for i in range(len(prompts))]
    if images:
        for image in images:
            content = await image.read()
            index = int(image.filename)
            image_contents[index].append(np.array(Image.open(io.BytesIO(content))))
    texts, logprobs, nb_input_token, nb_output_tokens = generate(
                                                    prompts=prompts,
                                                    systems=systems,
                                                    model_name=model_name,
                                                    temperature=temperature,
                                                    images=image_contents,
                                                    json_format=json_format
                                                )

    logprobs = replace_inf(data=logprobs,
                           replacement=-10000)

    co2 = tracker.stop()
    impact = co2*1.65
    data = {
        "texts": texts,
        "logprobs": logprobs,
        "nb_input_tokens": nb_input_token,
        "nb_output_tokens": nb_output_tokens,
        "impacts": [co2, co2, " gCO2eq"],
        "energy": [impact, impact, " Wh"]
    }
    return data


@app.post("/reranking")
async def reranking(request: RerankingBase):
    loadmodel(model_name=request.model_name)
    
    tracker = EmissionsTracker(project_name="vllm-inference",
                               save_to_file=False)
    tracker.start()
    
    if not request.contexts:
        raise HTTPException(
            status_code=400, detail="The contexts list should not be empty."
        )
    if not request.model_name:
        raise HTTPException(
            status_code=400, detail="The model name should not be None."
        )
    if not request.query:
        raise HTTPException(status_code=400, detail="The query should not be None.")

    scores, nb_input_tokens = rerank(
        query=request.query, contexts=request.contexts, model_name=request.model_name
    )
    co2 = tracker.stop()
    impact = co2*1.65
    data = {"scores": scores, 
            "nb_input_tokens": nb_input_tokens,
            "impacts": [co2, co2, " gCO2eq"],
            "energy": [impact, impact, " Wh"]}
    return data


@app.post("/setHFkey")
def setHFkey(request: HFKey):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = request.key