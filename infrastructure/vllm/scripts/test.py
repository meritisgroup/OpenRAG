import requests
import lorem

base_url = "http://127.0.0.1:8000"
url = base_url + "/load_model"
data = {
    "model_name": "Linq-AI-Research/Linq-Embed-Mistral",
    "gpu_memory_utilization": 0.5,
}

# response = requests.post(url, json=data)

nb_embeddings = 1
length = 2
texts = []
for i in range(nb_embeddings):
    temp = ""
    for j in range(length):
        temp += lorem.paragraph()
    texts.append(temp)

model = "mixedbread-ai/mxbai-embed-large-v1"
# model = "Linq-AI-Research/Linq-Embed-Mistral"

url = base_url + "/embeddings"
data = {"model_name": model, "texts": texts}

response = requests.post(url, json=data)
print(response.json())

query = "Quelle est la capital de la France?"
texts = [
    "La capitale de la france est Paris.",
    "La capitale de la france est Marseille.",
    "Mon chien s'appelle ",
    "La capital francaise est Paris.",
    "La capital de la belgique est bruxelle.",
]
model = "BAAI/bge-reranker-v2-m3"
url = base_url + "/reranking"
data = {"model_name": model, "contexts": texts, "query": query}
response = requests.post(url, json=data)
print(response.json())


prompts = [
    "Quelle est la capital de la France?",
    "Qui est le pdf de total?",
    "Peux tu me dire ce qu'est le deep learning?",
    "Quelle est la capital de la France?",
    "Qui est le pdf de total?",
    "Peux tu me dire ce qu'est le deep learning?",
    "Quelle est la capital de la France?",
    "Qui est le pdf de total?",
    "Peux tu me dire ce qu'est le deep learning?",
    "Quelle est la capital de la France?",
    "Qui est le pdf de total?",
    "Peux tu me dire ce qu'est le deep learning?",
]
systems = [
    "You are an AI assitant which answer correctly and honestly to the given question.",
    "You are an AI assitant which answer correctly and honestly to the given question.",
    "You are an AI assitant which answer correctly and honestly to the given question.",
    "You are an AI assitant which answer correctly and honestly to the given question.",
    "You are an AI assitant which answer correctly and honestly to the given question.",
    "You are an AI assitant which answer correctly and honestly to the given question.",
    "You are an AI assitant which answer correctly and honestly to the given question.",
    "You are an AI assitant which answer correctly and honestly to the given question.",
    "You are an AI assitant which answer correctly and honestly to the given question.",
    "You are an AI assitant which answer correctly and honestly to the given question.",
    "You are an AI assitant which answer correctly and honestly to the given question.",
    "You are an AI assitant which answer correctly and honestly to the given question.",
]
"""
url = base_url+"/load_model"
data = {"model_name": "google/gemma-2-2b-it",
         "prompts": prompts,
         "systems": systems}

response = requests.post(url, json=data)

url = base_url+"/load_model"
data = {"model_name": "microsoft/phi-4",
         "prompts": prompts,
         "systems": systems}
response = requests.post(url, json=data)

url = base_url+"/load_model"
data = {"model_name": "mixedbread-ai/mxbai-embed-large-v1",
         "prompts": prompts,
         "systems": systems}
response = requests.post(url, json=data)

url = base_url+"/release_memory"
response = requests.post(url,
                         json={})

url = base_url+"/load_model"
data = {"model_name": "Linq-AI-Research/Linq-Embed-Mistral",
         "prompts": prompts,
         "systems": systems}
response = requests.post(url, json=data)


url = base_url+"/load_model"
data = {"model_name": "meta-llama/Llama-3.1-8B-Instruct",
         "prompts": prompts,
         "systems": systems}
response = requests.post(url, json=data)
#print(response.json())


url = base_url+"/release_memory"
response = requests.post(url,
                         json={})
"""
