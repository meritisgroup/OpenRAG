import numpy as np
from .prompts import prompts


def run_batch_contextual(agent, doc_chunks, doc_content, language="EN"):
    chunk_with_context = []
    user_prompts = []
    system_prompts = []
    for chunk in doc_chunks:
        prompt = (
            prompts[language]["generate_context"]["QUERY_TEMPLATE"]
            .replace("{CHUNK_CONTENT}", chunk.text)
            .replace("{WHOLE_DOCUMENT}", doc_content)
        )
        system_prompt = prompts[language]["generate_context"]["SYSTEM_PROMPT"]
        user_prompts.append(prompt)
        system_prompts.append(system_prompt)

    taille_batch = 100
    contexts = None
    for i in range(0, len(user_prompts), taille_batch):
        results = agent.multiple_predict(prompts=user_prompts[i:i + taille_batch],
                                         system_prompts=system_prompts[i:i + taille_batch])
        if contexts is None:
           contexts = results
        else:
           contexts["texts"] = contexts["texts"] + results["texts"]
           contexts["nb_input_tokens"]+=np.sum(results["nb_input_tokens"])
           contexts["nb_output_tokens"]+=np.sum(results["nb_output_tokens"])
           contexts["impacts"][0]+=results["impacts"][0]
           contexts["impacts"][1]+=results["impacts"][1]
           contexts["energy"][0]+=results["energy"][0]
           contexts["energy"][1]+=results["energy"][1]
    
    nb_output_tokens = np.sum(contexts["nb_output_tokens"])
    nb_input_tokens = np.sum(contexts["nb_input_tokens"])
    for i in range(len(contexts["texts"])):
        context = contexts["texts"][i]
        new_chunk = f"Chunk context: {context} \n ------ \n Chunk: {doc_chunks[i].text}"
        chunk_with_context.append(new_chunk)
    return {
            "texts": chunk_with_context,
            "nb_output_tokens": nb_output_tokens,
            "nb_input_tokens": nb_input_tokens,
            "impacts": contexts["impacts"],
            "energy": contexts["energy"]
        }


def run_serial_contextual(agent, doc_chunks, doc_content, language="EN"):
    chunk_with_context = []
    nb_input_tokens = 0
    nb_output_tokens = 0
    energy = None
    impacts = None
    for chunk in doc_chunks:
        user_prompt = (
            prompts[language]["generate_context"]["QUERY_TEMPLATE"]
            .replace("{CHUNK_CONTENT}", chunk.text)
            .replace("{WHOLE_DOCUMENT}", doc_content)
        )
        system_prompt = prompts[language]["generate_context"]["SYSTEM_PROMPT"]
        context = agent.predict(prompt=user_prompt, system_prompt=system_prompt)
        nb_input_tokens += np.sum(context["nb_input_tokens"])
        nb_output_tokens += np.sum(context["nb_output_tokens"])

        if energy is None:
            energy = context["energy"]
            impacts = context["impacts"]
        else:
            energy[0] += context["energy"][0]
            impacts[0] += context["impacts"][0]
            energy[1] += context["energy"][1]
            impacts[1] += context["impacts"][1]

        c = context["texts"][0]
        new_chunk = f"Chunk context: {c} \n Chunk: {chunk.text}"
        chunk_with_context.append(new_chunk)
    return {
        "texts": chunk_with_context,
        "nb_output_tokens": nb_input_tokens,
        "nb_input_tokens": nb_output_tokens,
        "energy": energy,
        "impacts": impacts
    }
