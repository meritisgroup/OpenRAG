from ..naive_rag.query import NaiveSearch
from ..naive_rag.agent import NaiveRagAgent
from .prompts import prompts
from ..naive_rag.indexation import contexts_to_prompts
import numpy as np
from backend.database.rag_classes import Chunk
from ...utils.chunk_lists_merger import merge_chunk_lists


class SelfRagAgent(NaiveRagAgent):

    def __init__(
        self, config_server: dict, models_infos: dict, dbs_name: list[str], data_folders_name: list[str]
    ) -> None:

        super().__init__(
            config_server=config_server,
            models_infos=models_infos,
            dbs_name=dbs_name,
            data_folders_name=data_folders_name,
        )
        self.language = config_server["language"]
        self.prompts = prompts[self.language]


    def get_nb_token_embeddings(self):
        return self.data_manager.get_nb_token_embeddings()

    def get_rag_context(self, query: str, nb_chunks: int = 5) -> list[list[Chunk]]:
        """
        Takes a query and retrieves a given number of chunks using the NaiveSearch implemented

        Args:
            query (str) : The query that needs answering
            nb_chunks (int) : Number of chunks to be retrieved
        Output:
            context (list[str]) : All retrieved chunks
        """
        ns = NaiveSearch(data_manager=self.data_manager, nb_chunks=nb_chunks)
        chunk_lists = ns.get_context(query=query)
        return chunk_lists

    def __run_batch_answer(self, query, agent, 
                           chunk_lists: list[list[Chunk]],
                           options_generation = None):

        chunk_list = merge_chunk_lists(chunk_lists)
        useful_chunks = []
        prompts = []
        system_prompts = []
        impacts, energies = [0, 0, ""], [0, 0, ""]
        for i, context in enumerate(chunk_list):
            prompt = self.prompts["document_relevance"]["QUERY_TEMPLATE"].format(
                context=chunk_list[i].text, query=query
            )
            system_prompt = self.prompts["document_relevance"]["SYSTEM_PROMPT"]
            prompts.append(prompt)
            system_prompts.append(system_prompt)
        scores = agent.multiple_predict(prompts=prompts,
                                        model=self.llm_model,
                                        system_prompts=system_prompts)
        impacts[2] = scores["impacts"][2]
        impacts[0] += scores["impacts"][0]
        impacts[1] += scores["impacts"][1]
        energies[2] = scores["energy"][2]
        energies[0] += scores["energy"][0]
        energies[1] += scores["energy"][1]

        nb_input_tokens = np.sum(scores["nb_input_tokens"])
        nb_output_tokens = np.sum(scores["nb_output_tokens"])

        scores = scores["texts"]
        final_answer = ""
        for j in range(len(chunk_list)):
            if "relevant" in scores[j].lower():
                useful_chunks.append(chunk_list[j])

        if len(useful_chunks) > 0:
            answers_fully = []
            answers_partially = []
            prompts = []
            system_prompts = []
            for i, chunk in enumerate(useful_chunks):
                prompt = self.prompts["smooth_generation"]["QUERY_TEMPLATE"].format(
                    context=chunk.text, query=query
                )
                system_prompt = self.prompts["smooth_generation"]["SYSTEM_PROMPT"]
                prompts.append(prompt)
                system_prompts.append(system_prompt)
            answers = agent.multiple_predict(prompts=prompts, 
                                             system_prompts=system_prompts,
                                             model=self.llm_model)
            
            nb_input_tokens += np.sum(answers["nb_input_tokens"])
            nb_output_tokens += np.sum(answers["nb_output_tokens"])

            impacts[0] += answers["impacts"][0]
            impacts[1] += answers["impacts"][1]

            energies[0] += answers["energy"][0]
            energies[1] += answers["energy"][1]
            answers = answers["texts"]
            prompts = []
            system_prompts = []
            for i, chunk in enumerate(useful_chunks):
                prompt = self.prompts["supported_generation"]["QUERY_TEMPLATE"].format(
                    context=chunk.text, query=answers[i]
                )
                system_prompt = self.prompts["supported_generation"]["SYSTEM_PROMPT"]
                prompts.append(prompt)
                system_prompts.append(system_prompt)
            supports = agent.multiple_predict(prompts=prompts,
                                              system_prompts=system_prompts,
                                              model=self.llm_model)

            impacts[0] += supports["impacts"][0]
            impacts[1] += supports["impacts"][1]
            energies[0] += supports["energy"][0]
            energies[1] += supports["energy"][1]

            nb_input_tokens += np.sum(supports["nb_input_tokens"])
            nb_output_tokens += np.sum(supports["nb_output_tokens"])

            supports = supports["texts"]
            prompts = []
            system_prompts = []
            for i, chunk in enumerate(useful_chunks):
                prompt = self.prompts["rate_generation"]["QUERY_TEMPLATE"].format(
                    context=answers[i], query=query
                )
                system_prompt = self.prompts["rate_generation"]["SYSTEM_PROMPT"]
                prompts.append(prompt)
                system_prompts.append(system_prompt)
            rates = agent.multiple_predict(prompts=prompts,
                                           system_prompts=system_prompts,
                                           model=self.llm_model)
            impacts[0] += rates["impacts"][0]
            impacts[1] += rates["impacts"][1]
            energies[0] += rates["energy"][0]
            energies[1] += rates["energy"][1]
            nb_input_tokens += np.sum(rates["nb_input_tokens"])
            nb_output_tokens += np.sum(rates["nb_output_tokens"])
            rates = rates["texts"]
            for i, support in enumerate(supports):
                if "fully supported" in support.lower():
                    answers_fully.append([answers[i], rates[i]])
                elif "partially supported" in support.lower():
                    answers_partially.append([answers[i], rates[i]])

            best_rate = 0
            if len(answers_fully) > 0:
                for i in range(len(answers_fully)):
                    if int(answers_fully[i][1]) > best_rate:
                        best_rate = int(answers_fully[i][1])
                        final_answer = answers_fully[i][0]
                        context = useful_chunks[i]
            elif len(answers_partially) > 0:
                for i in range(len(answers_partially)):
                    if int(answers_partially[i][1]) > best_rate:
                        best_rate = int(answers_partially[i][1])
                        answer = answers_partially[i][0]
                        context = useful_chunks[i]
            else:
                prompt = self.prompts["conversationnal"]["QUERY_TEMPLATE"].format(
                    query=query
                )
                answer = agent.predict(prompt=prompt,
                                       system_prompt=self.system_prompt,
                                       model=self.llm_model)
                nb_input_tokens += np.sum(answer["nb_input_tokens"])
                nb_output_tokens += np.sum(answer["nb_output_tokens"])

                impacts[0] += answer["impacts"][0]
                impacts[1] += answer["impacts"][1]

                energies[0] += answer["energy"][0]
                energies[1] += answer["energy"][1]
                answer = answer["texts"]
                context = []

        else:
            prompt = self.prompts["conversationnal"]["QUERY_TEMPLATE"].format(
                query=query
            )

            answer = agent.predict(prompt=prompt,
                                   system_prompt=self.system_prompt,
                                   options_generation=options_generation,
                                   model=self.llm_model)
            nb_input_tokens += np.sum(answer["nb_input_tokens"])
            nb_output_tokens += np.sum(answer["nb_output_tokens"])

            impacts[0] += answer["impacts"][0]
            impacts[1] += answer["impacts"][1]

            energies[0] += answer["energy"][0]
            energies[1] += answer["energy"][1]
            answer = answer["texts"]
            context = []
        return {
            "texts": final_answer,
            "context": context,
            "nb_input_tokens": nb_input_tokens,
            "nb_output_tokens": nb_output_tokens,
            "impacts": impacts,
            "energy": energies,
        }

    def generate_answer(
        self,
        query: str,
        model: str = None,
        nb_chunks: str = 5,
        batch: bool = True,
        options_generation = None
    ) -> str:
        """Generate an answer to the query"""
        if options_generation is None:
            options_generation = self.config_server["options_generation"]

        nb_input_tokens = 0
        nb_output_tokens = 0
        impacts, energies = [0, 0, ""], [0, 0, ""]
        if self.reformulate_query:
            query, input_t, ouput_t, impacts, energies = self.reformulater.reformulate(
                query=query, nb_reformulation=1
            )
            query = query[0]
            nb_input_tokens += np.sum(input_t)
            nb_output_tokens += np.sum(ouput_t)
        agent = self.agent

        contexts = ""
        prompt = self.prompts["retrieval_necessary"]["QUERY_TEMPLATE"].format(
            query=query
        )

        system_prompt = self.prompts["retrieval_necessary"]["SYSTEM_PROMPT"]
        retrieval_necessary = agent.predict(prompt=prompt,
                                            system_prompt=system_prompt,
                                            model=self.llm_model)

        nb_input_tokens += np.sum(retrieval_necessary["nb_input_tokens"])
        nb_output_tokens += np.sum(retrieval_necessary["nb_output_tokens"])
        impacts[2] = retrieval_necessary["impacts"][2]
        impacts[0] += retrieval_necessary["impacts"][0]
        impacts[1] += retrieval_necessary["impacts"][1]
        energies[2] = retrieval_necessary["energy"][2]
        energies[0] += retrieval_necessary["energy"][0]
        energies[1] += retrieval_necessary["energy"][1]

        retrieval_necessary = retrieval_necessary["texts"]

        if "yes" in retrieval_necessary.lower():
            chunk_lists = self.get_rag_context(query=query, nb_chunks=nb_chunks)


            answer = self.__run_batch_answer(
                                    query=query,
                                    agent=agent, 
                                    chunk_lists=chunk_lists,
                                    options_generation = options_generation
                                )
            nb_input_tokens += np.sum(answer["nb_input_tokens"])
            nb_output_tokens += np.sum(answer["nb_output_tokens"])
            context = answer["context"]

        else:
            prompt = self.prompts["conversationnal"]["QUERY_TEMPLATE"].format(
                query=query
            )
            answer = agent.predict(prompt=prompt,
                                   system_prompt=self.system_prompt,
                                   options_generation=options_generation,
                                   model=self.llm_model)
            context = []
            nb_input_tokens += np.sum(answer["nb_input_tokens"])
            nb_output_tokens += np.sum(answer["nb_output_tokens"])

        impacts[2] = answer["impacts"][2]
        impacts[0] += answer["impacts"][0]
        impacts[1] += answer["impacts"][1]
        energies[2] = answer["energy"][2]
        energies[0] += answer["energy"][0]
        energies[1] += answer["energy"][1]

        if type(context)!=list and type(context)!=np.ndarray:
            context = [context]
            
        return {
            "answer": answer["texts"],
            "nb_input_tokens": nb_input_tokens,
            "nb_output_tokens": nb_output_tokens,
            "context": context,
            "impacts": impacts,
            "energy": energies,
            "original_query": query
        }

    def release_gpu_memory(self):
        self.agent.release_memory()
