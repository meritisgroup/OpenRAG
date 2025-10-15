import numpy as np
from .prompts import prompts
from ...utils.progress import ProgressBar


def get_neighbors_chunks(doc_chunks, index, max_context_len=8192):
    local_context = ""
    left, right = index - 1, index + 1
    while len(local_context) < max_context_len and (left >= 0 or right < len(doc_chunks)):
        candidate_context = local_context

        if left >= 0:
            candidate_context = doc_chunks[left].text + "\n" + candidate_context
            left -= 1

        if right < len(doc_chunks):
            candidate_context = candidate_context + "\n" + doc_chunks[right].text
            right += 1

        if len(candidate_context) <= max_context_len:
            local_context = candidate_context
        else:
            break
    return local_context


def run_contextual(agent, doc_chunks, doc_content, language="EN"):
    chunk_with_context = []
    user_prompts = []
    system_prompts = []
    for i in range(len(doc_chunks)):
        context_chunk = get_neighbors_chunks(doc_chunks=doc_chunks,
                                             index=i,
                                             max_context_len=8192)
        prompt = (
            prompts[language]["generate_context"]["QUERY_TEMPLATE"]
            .replace("{CHUNK_CONTENT}", doc_chunks[i].text)
            .replace("{WHOLE_DOCUMENT}", context_chunk)
        )
        system_prompt = prompts[language]["generate_context"]["SYSTEM_PROMPT"]
        user_prompts.append(prompt)
        system_prompts.append(system_prompt)

    taille_batch = 500
    contexts = None
    range_chunks = range(0, len(user_prompts), taille_batch)
    progress_bar_chunks = ProgressBar(total=len(range_chunks))
    j = 0
    for i in range_chunks:
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
        progress_bar_chunks.update(j)
        j+=1
    progress_bar_chunks.clear()
        
    nb_output_tokens = np.sum(contexts["nb_output_tokens"])
    nb_input_tokens = np.sum(contexts["nb_input_tokens"])
    for i in range(len(contexts["texts"])):
        context = contexts["texts"][i]
        new_chunk = f"Chunk context:\n {context} \n\nChunk:\n {doc_chunks[i].text}"
        chunk_with_context.append(new_chunk)
    return {
            "texts": chunk_with_context,
            "nb_output_tokens": nb_output_tokens,
            "nb_input_tokens": nb_input_tokens,
            "impacts": contexts["impacts"],
            "energy": contexts["energy"]
        }