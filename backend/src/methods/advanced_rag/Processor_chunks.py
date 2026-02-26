from methods.contextual_retrieval_rag.contextual import run_contextual
from database.rag_classes import Chunk
from .prompts import prompts
import numpy as np

class Processor_chunks:

    def __init__(self, agent, type_processor_chunks=[], language='EN'):
        self.agent = agent
        self.type_processor_chunks = type_processor_chunks
        if type(self.type_processor_chunks) == str:
            self.type_processor_chunks = [self.type_processor_chunks]
        self.language = language

    def process_chunk(self, chunks, doc_content, model):
        data = {'chunks': chunks, 'nb_output_tokens': 0, 'nb_input_tokens': 0}
        if len(self.type_processor_chunks) == 0:
            return data
        first_process = True
        if 'Global_sum_up' in self.type_processor_chunks:
            data_temp = self.run_global_sum_up(chunks=data['chunks'], doc_content=doc_content, model=model)
            data['chunks'] = data_temp['chunks']
            data['nb_output_tokens'] += data_temp['nb_output_tokens']
            data['nb_input_tokens'] += data_temp['nb_input_tokens']
            first_process = False
        if 'Extractor_metadata' in self.type_processor_chunks:
            data_temp = self.run_extractor_metadata(chunks=data['chunks'], doc_content=doc_content, model=model, first_process=first_process)
            data['chunks'] = data_temp['chunks']
            data['nb_output_tokens'] += data_temp['nb_output_tokens']
            data['nb_input_tokens'] += data_temp['nb_input_tokens']
            first_process = False
        if 'Contextual' in self.type_processor_chunks:
            data_temp = self.run_contextual(chunks=data['chunks'], doc_content=doc_content, model=model, first_process=first_process)
            data['chunks'] = data_temp['chunks']
            data['nb_output_tokens'] += data_temp['nb_output_tokens']
            data['nb_input_tokens'] += data_temp['nb_input_tokens']
            first_process = False
        return data

    def run_global_sum_up(self, chunks, doc_content, model):
        data = run_global_sum_up(chunks=chunks, doc_content=doc_content, model=model, agent=self.agent, language=self.language)
        results = {'chunks': [], 'nb_output_tokens': data['nb_output_tokens'], 'nb_input_tokens': data['nb_input_tokens']}
        for i in range(len(chunks)):
            results['chunks'].append(Chunk(text=data['texts'][i], document=chunks[i].document, id=chunks[i].id))
        return results

    def run_extractor_metadata(self, chunks, doc_content, model, first_process):
        data = run_extract_metadata(chunks=chunks, doc_content=doc_content, model=model, agent=self.agent, language=self.language, first_process=first_process)
        results = {'chunks': [], 'nb_output_tokens': data['nb_output_tokens'], 'nb_input_tokens': data['nb_input_tokens']}
        for i in range(len(chunks)):
            results['chunks'].append(Chunk(text=data['texts'][i], document=chunks[i].document, id=chunks[i].id))
        return results

    def run_contextual(self, chunks, doc_content, model, first_process=True):
        data = run_contextual(agent=self.agent, doc_chunks=chunks, model=model, doc_content=doc_content, language=self.language, first_process=first_process)
        results = {'chunks': [], 'nb_output_tokens': data['nb_output_tokens'], 'nb_input_tokens': data['nb_input_tokens']}
        for i in range(len(chunks)):
            results['chunks'].append(Chunk(text=data['texts'][i], document=chunks[i].document, id=chunks[i].id))
        return results

def split_string(s: str, max_len: int=96000) -> list[str]:
    return [s[i:i + max_len] for i in range(0, len(s), max_len)]

def run_extract_metadata(chunks, agent, model, doc_content, language, first_process):
    chunks_resume = split_string(doc_content)[0]
    system_prompt = prompts[language]['extract_metadata']['SYSTEM_PROMPT']
    prompt_metadata = prompts[language]['extract_metadata']['QUERY_TEMPLATE'].replace('{WHOLE_DOCUMENT}', chunks_resume)
    context = agent.predict(prompt=prompt_metadata, system_prompt=system_prompt, model=model)
    nb_output_tokens = np.sum(context['nb_output_tokens'])
    nb_input_tokens = np.sum(context['nb_input_tokens'])
    impacts = context['impacts']
    energy = context['energy']
    chunk_with_metadata = []
    context_text = context['texts']
    for i in range(len(chunks)):
        if first_process:
            new_chunk = f'<Document_metadata>\n {context_text}\n</Document_metadata> \n\n<Chunk>:\n {chunks[i].text}\n</Chunk>'
        else:
            new_chunk = f'<Document_metadata>\n {context_text}\n</Document_metadata> \n\n {chunks[i].text}'
        chunk_with_metadata.append(new_chunk)
    return {'texts': chunk_with_metadata, 'nb_output_tokens': nb_output_tokens, 'nb_input_tokens': nb_input_tokens, 'impacts': impacts, 'energy': energy}

def run_global_sum_up(chunks, agent, model, doc_content, language):
    chunks_resume = split_string(doc_content)[:1]
    prompts_resume = [prompts[language]['generate_global_sum']['QUERY_TEMPLATE'].replace('{WHOLE_DOCUMENT}', chunk) for chunk in chunks_resume]
    system_prompt = prompts[language]['generate_global_sum']['SYSTEM_PROMPT']
    contexts = agent.multiple_predict(prompts=prompts_resume, system_prompt=system_prompt, model=model)
    nb_output_tokens = np.sum(contexts['nb_output_tokens'])
    nb_input_tokens = np.sum(contexts['nb_input_tokens'])
    impacts = contexts['impacts']
    energy = contexts['energy']
    if len(chunks_resume) > 1:
        system_prompt = prompts[language]['resume_of_resume']['SYSTEM_PROMPT']
        prompt = ''
        for i in range(len(contexts['texts'])):
            prompt += '<resume>\n{}</resume>\n\n'.format(contexts['texts'][i])
        contexts = agent.predict(prompt=prompt, system_prompt=system_prompt, model=model)
        nb_output_tokens += np.sum(contexts['nb_output_tokens'])
        nb_input_tokens += np.sum(contexts['nb_input_tokens'])
        impacts[0] += impacts[0]
        impacts[1] += impacts[1]
        energy[0] += energy[0]
        energy[1] += energy[1]
        context_text = contexts['texts']
    else:
        context_text = contexts['texts'][0]
    chunk_with_context = []
    for i in range(len(chunks)):
        new_chunk = f'<Document_sum_up>\n {context_text}\n</Document_sum_up> \n\n<Chunk>:\n {chunks[i].text}\n</Chunk>'
        chunk_with_context.append(new_chunk)
    return {'texts': chunk_with_context, 'nb_output_tokens': nb_output_tokens, 'nb_input_tokens': nb_input_tokens, 'impacts': impacts, 'energy': energy}