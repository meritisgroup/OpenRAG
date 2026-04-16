from methods.naive_rag.query import NaiveSearch
from methods.naive_rag.agent import NaiveRagAgent
from .prompts import prompts
from methods.naive_rag.indexation import contexts_to_prompts
import numpy as np
from database.rag_classes import Chunk
from utils.chunk_lists_merger import merge_chunk_lists

class SelfRagAgent(NaiveRagAgent):

    def __init__(self, config_server: dict, models_infos: dict, dbs_name: list[str], data_folders_name: list[str]) -> None:
        super().__init__(config_server=config_server, models_infos=models_infos, dbs_name=dbs_name, data_folders_name=data_folders_name)
        self.language = config_server['language']
        self.prompts = prompts[self.language]

    def get_nb_token_embeddings(self):
        return self.data_manager.get_nb_token_embeddings()

    def get_rag_context(self, query: str, nb_chunks: int=5) -> list[list[Chunk]]:
        ns = NaiveSearch(data_manager=self.data_manager, nb_chunks=nb_chunks)
        chunk_lists = ns.get_context(query=query)
        return chunk_lists

    def __score_relevance(self, agent, chunk_list, query, prompts, system_prompt):
        scores = agent.multiple_predict(prompts=prompts, model=self.llm_model, system_prompt=system_prompt)
        impacts = [scores['impacts'][0], scores['impacts'][1], scores['impacts'][2]]
        energies = [scores['energy'][0], scores['energy'][1], scores['energy'][2]]
        nb_input_tokens = np.sum(scores['nb_input_tokens'])
        nb_output_tokens = np.sum(scores['nb_output_tokens'])
        scores = scores['texts']
        useful_chunks = []
        for j in range(len(chunk_list)):
            if 'relevant' in scores[j].lower():
                useful_chunks.append(chunk_list[j])
        return {
            'useful_chunks': useful_chunks,
            'nb_input_tokens': nb_input_tokens,
            'nb_output_tokens': nb_output_tokens,
            'impacts': impacts,
            'energy': energies,
        }

    def __generate_answers(self, agent, useful_chunks, query):
        prompts = []
        for (i, chunk) in enumerate(useful_chunks):
            prompt = self.prompts['smooth_generation']['QUERY_TEMPLATE'].format(context=chunk.text, query=query)
            prompts.append(prompt)
        system_prompt = self.prompts['smooth_generation']['SYSTEM_PROMPT']
        answers = agent.multiple_predict(prompts=prompts, system_prompt=system_prompt, model=self.llm_model)
        nb_input_tokens = np.sum(answers['nb_input_tokens'])
        nb_output_tokens = np.sum(answers['nb_output_tokens'])
        impacts = [answers['impacts'][0], answers['impacts'][1], 0]
        energies = [answers['energy'][0], answers['energy'][1], 0]
        answers = answers['texts']
        return {
            'answers': answers,
            'nb_input_tokens': nb_input_tokens,
            'nb_output_tokens': nb_output_tokens,
            'impacts': impacts,
            'energy': energies,
        }

    def __check_support(self, agent, useful_chunks, answers):
        prompts = []
        for (i, chunk) in enumerate(useful_chunks):
            prompt = self.prompts['supported_generation']['QUERY_TEMPLATE'].format(context=chunk.text, query=answers[i])
            prompts.append(prompt)
        system_prompt = self.prompts['supported_generation']['SYSTEM_PROMPT']
        supports = agent.multiple_predict(prompts=prompts, system_prompt=system_prompt, model=self.llm_model)
        nb_input_tokens = np.sum(supports['nb_input_tokens'])
        nb_output_tokens = np.sum(supports['nb_output_tokens'])
        impacts = [supports['impacts'][0], supports['impacts'][1], 0]
        energies = [supports['energy'][0], supports['energy'][1], 0]
        supports = supports['texts']
        return {
            'supports': supports,
            'nb_input_tokens': nb_input_tokens,
            'nb_output_tokens': nb_output_tokens,
            'impacts': impacts,
            'energy': energies,
        }

    def __rate_answers(self, agent, answers, query):
        prompts = []
        for (i, chunk) in enumerate(answers):
            prompt = self.prompts['rate_generation']['QUERY_TEMPLATE'].format(context=answers[i], query=query)
            prompts.append(prompt)
        system_prompt = self.prompts['rate_generation']['SYSTEM_PROMPT']
        rates = agent.multiple_predict(prompts=prompts, system_prompt=system_prompt, model=self.llm_model)
        nb_input_tokens = np.sum(rates['nb_input_tokens'])
        nb_output_tokens = np.sum(rates['nb_output_tokens'])
        impacts = [rates['impacts'][0], rates['impacts'][1], 0]
        energies = [rates['energy'][0], rates['energy'][1], 0]
        rates = rates['texts']
        return {
            'rates': rates,
            'nb_input_tokens': nb_input_tokens,
            'nb_output_tokens': nb_output_tokens,
            'impacts': impacts,
            'energy': energies,
        }

    def __select_best_answer(self, agent, answers, supports, rates, useful_chunks, query, options_generation):
        answers_fully = []
        answers_partially = []
        fully_indices = []
        partially_indices = []
        for (i, support) in enumerate(supports):
            if 'fully supported' in support.lower():
                answers_fully.append([answers[i], rates[i]])
                fully_indices.append(i)
            elif 'partially supported' in support.lower():
                answers_partially.append([answers[i], rates[i]])
                partially_indices.append(i)
        final_answer = ''
        best_rate = 0
        context = None
        if len(answers_fully) > 0:
            for i in range(len(answers_fully)):
                if int(answers_fully[i][1]) > best_rate:
                    best_rate = int(answers_fully[i][1])
                    final_answer = answers_fully[i][0]
                    context = useful_chunks[fully_indices[i]]
        elif len(answers_partially) > 0:
            for i in range(len(answers_partially)):
                if int(answers_partially[i][1]) > best_rate:
                    best_rate = int(answers_partially[i][1])
                    final_answer = answers_partially[i][0]
                    context = useful_chunks[partially_indices[i]]
        else:
            prompt = self.prompts['conversationnal']['QUERY_TEMPLATE'].format(query=query)
            answer = agent.predict(prompt=prompt, system_prompt=self.system_prompt, model=self.llm_model)
            nb_input_tokens = np.sum(answer['nb_input_tokens'])
            nb_output_tokens = np.sum(answer['nb_output_tokens'])
            impacts = [answer['impacts'][0], answer['impacts'][1], 0]
            energies = [answer['energy'][0], answer['energy'][1], 0]
            answer = answer['texts']
            context = []
            return {
                'texts': '',
                'context': context,
                'nb_input_tokens': nb_input_tokens,
                'nb_output_tokens': nb_output_tokens,
                'impacts': impacts,
                'energy': energies,
            }
        return {
            'texts': final_answer,
            'context': context,
            'nb_input_tokens': 0,
            'nb_output_tokens': 0,
            'impacts': [0, 0, 0],
            'energy': [0, 0, 0],
        }

    def __run_batch_answer(self, query, agent, chunk_lists: list[list[Chunk]], options_generation=None):
        chunk_list = merge_chunk_lists(chunk_lists)
        prompts = []
        for (i, context) in enumerate(chunk_list):
            prompt = self.prompts['document_relevance']['QUERY_TEMPLATE'].format(context=chunk_list[i].text, query=query)
            prompts.append(prompt)
        system_prompt = self.prompts['document_relevance']['SYSTEM_PROMPT']
        relevance = self.__score_relevance(agent, chunk_list, query, prompts, system_prompt)
        nb_input_tokens = relevance['nb_input_tokens']
        nb_output_tokens = relevance['nb_output_tokens']
        impacts = relevance['impacts']
        energies = relevance['energy']
        useful_chunks = relevance['useful_chunks']
        if len(useful_chunks) > 0:
            gen = self.__generate_answers(agent, useful_chunks, query)
            nb_input_tokens += gen['nb_input_tokens']
            nb_output_tokens += gen['nb_output_tokens']
            impacts[0] += gen['impacts'][0]
            impacts[1] += gen['impacts'][1]
            energies[0] += gen['energy'][0]
            energies[1] += gen['energy'][1]
            answers = gen['answers']
            sup = self.__check_support(agent, useful_chunks, answers)
            nb_input_tokens += sup['nb_input_tokens']
            nb_output_tokens += sup['nb_output_tokens']
            impacts[0] += sup['impacts'][0]
            impacts[1] += sup['impacts'][1]
            energies[0] += sup['energy'][0]
            energies[1] += sup['energy'][1]
            supports = sup['supports']
            rat = self.__rate_answers(agent, answers, query)
            nb_input_tokens += rat['nb_input_tokens']
            nb_output_tokens += rat['nb_output_tokens']
            impacts[0] += rat['impacts'][0]
            impacts[1] += rat['impacts'][1]
            energies[0] += rat['energy'][0]
            energies[1] += rat['energy'][1]
            rates = rat['rates']
            result = self.__select_best_answer(agent, answers, supports, rates, useful_chunks, query, options_generation)
            nb_input_tokens += result['nb_input_tokens']
            nb_output_tokens += result['nb_output_tokens']
            impacts[0] += result['impacts'][0]
            impacts[1] += result['impacts'][1]
            energies[0] += result['energy'][0]
            energies[1] += result['energy'][1]
            final_answer = result['texts']
            context = result['context']
        else:
            prompt = self.prompts['conversationnal']['QUERY_TEMPLATE'].format(query=query)
            answer = agent.predict(prompt=prompt, system_prompt=self.system_prompt, options_generation=options_generation, model=self.llm_model)
            nb_input_tokens += np.sum(answer['nb_input_tokens'])
            nb_output_tokens += np.sum(answer['nb_output_tokens'])
            impacts[0] += answer['impacts'][0]
            impacts[1] += answer['impacts'][1]
            energies[0] += answer['energy'][0]
            energies[1] += answer['energy'][1]
            answer = answer['texts']
            final_answer = ''
            context = []
        return {'texts': final_answer, 'context': context, 'nb_input_tokens': nb_input_tokens, 'nb_output_tokens': nb_output_tokens, 'impacts': impacts, 'energy': energies}

    def generate_answer(self, query: str, model: str=None, nb_chunks: int=5, batch: bool=True, options_generation=None) -> str:
        if options_generation is None:
            options_generation = self.config_server['options_generation']
        nb_input_tokens = 0
        nb_output_tokens = 0
        (impacts, energies) = ([0, 0, ''], [0, 0, ''])
        if self.reformulate_query:
            (query, input_t, output_t, impacts, energies) = self.reformulater.reformulate(query=query, nb_reformulation=1)
            query = query[0]
            nb_input_tokens += np.sum(input_t)
            nb_output_tokens += np.sum(output_t)
        agent = self.agent
        contexts = ''
        prompt = self.prompts['retrieval_necessary']['QUERY_TEMPLATE'].format(query=query)
        system_prompt = self.prompts['retrieval_necessary']['SYSTEM_PROMPT']
        retrieval_necessary = agent.predict(prompt=prompt, system_prompt=system_prompt, model=self.llm_model)
        nb_input_tokens += np.sum(retrieval_necessary['nb_input_tokens'])
        nb_output_tokens += np.sum(retrieval_necessary['nb_output_tokens'])
        impacts[2] = retrieval_necessary['impacts'][2]
        impacts[0] += retrieval_necessary['impacts'][0]
        impacts[1] += retrieval_necessary['impacts'][1]
        energies[2] = retrieval_necessary['energy'][2]
        energies[0] += retrieval_necessary['energy'][0]
        energies[1] += retrieval_necessary['energy'][1]
        retrieval_necessary = retrieval_necessary['texts']
        if 'yes' in retrieval_necessary.lower():
            chunk_lists = self.get_rag_context(query=query, nb_chunks=nb_chunks)
            answer = self.__run_batch_answer(query=query, agent=agent, chunk_lists=chunk_lists, options_generation=options_generation)
            nb_input_tokens += np.sum(answer['nb_input_tokens'])
            nb_output_tokens += np.sum(answer['nb_output_tokens'])
            context = answer['context']
        else:
            prompt = self.prompts['conversationnal']['QUERY_TEMPLATE'].format(query=query)
            answer = agent.predict(prompt=prompt, system_prompt=self.system_prompt, options_generation=options_generation, model=self.llm_model)
            context = []
            nb_input_tokens += np.sum(answer['nb_input_tokens'])
            nb_output_tokens += np.sum(answer['nb_output_tokens'])
        impacts[2] = answer['impacts'][2]
        impacts[0] += answer['impacts'][0]
        impacts[1] += answer['impacts'][1]
        energies[2] = answer['energy'][2]
        energies[0] += answer['energy'][0]
        energies[1] += answer['energy'][1]
        if type(context) != list and type(context) != np.ndarray:
            context = [context]
        return {'answer': answer['texts'], 'nb_input_tokens': nb_input_tokens, 'nb_output_tokens': nb_output_tokens, 'context': context, 'impacts': impacts, 'energy': energies, 'original_query': query}

    def release_gpu_memory(self):
        self.agent.release_memory()