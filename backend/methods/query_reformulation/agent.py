from ..advanced_rag.query import NaiveSearch
from ..naive_rag.agent import NaiveRagAgent
from itertools import chain
from .prompts import prompts
from .query_reformulation import query_reformulation
import numpy as np


class QueryReformulationRag(NaiveRagAgent):
    def __init__(
        self,
        config_server: dict,
        dbs_name: list[str],
        data_folders_name: list[str]
    ) -> None:

        super().__init__(
            config_server=config_server,
            dbs_name=dbs_name,
            data_folders_name=data_folders_name
        )
        self.nb_chunks = config_server["nb_chunks"]
        self.prompts = prompts[self.language]
        self.reformulater = query_reformulation(
            agent=self.agent, language=self.language
        )

    def get_nb_token_embeddings(self):
        return self.data_manager.get_nb_token_embeddings()

    def get_rag_context(
        self, query: str, nb_chunks: int = 5
    ) -> list[str]:
        ns = NaiveSearch(data_manager=self.data_manager,
                         nb_chunks=nb_chunks)
        context, docs_name = ns.get_context(query=query)
        return context, docs_name

    def generate_answer(
        self,
        query: str,
        nb_chunks: int = 5,
        nb_reformulation: int = 5,
        options_generation = None
    ) -> str:
        """Generate an answer to the query"""
        agent = self.agent

        queries, nb_input_tokens, nb_output_tokens, impact, energy = self.reformulater.reformulate(
            query=query, nb_reformulation=nb_reformulation
        )

        results = [
            self.get_rag_context(query=query, nb_chunks=nb_chunks) for query in queries
        ]
        contexts = []
        docs_name = []
        for result in results:
            contexts+=result[0]
            docs_name+=result[1]

        context, docs_name = self.contexts_to_prompts(contexts=contexts,
                                                      docs_name=docs_name)
        prompt = self.prompts["smooth_generation"]["QUERY_TEMPLATE"].format(
            context=context, query=query
        )
        system_prompt = self.prompts["smooth_generation"]["SYSTEM_PROMPT"]

        if options_generation is None:
            options_generation = self.config_server["options_generation"]

        answer = agent.predict(prompt=prompt, system_prompt=system_prompt,
                               options_generation=options_generation)
        nb_input_tokens += np.sum(answer["nb_input_tokens"])
        nb_output_tokens += np.sum(answer["nb_output_tokens"])

        impacts = answer["impacts"]
        impacts[0] += impact[0]
        impacts[1] += impact[1]

        energies = answer["energy"]
        energies[0] += energy[0]
        energies[1] += energy[1]

        return {
            "answer": answer["texts"],
            "nb_input_tokens": np.sum(nb_input_tokens),
            "nb_output_tokens": np.sum(nb_output_tokens),
            "context": context,
            "docs_name": docs_name,
            "impacts": impacts,
            "energy" : energies
        }

    def release_gpu_memory(self):
        self.agent.release_memory()

    def get_rag_contexts(
        self, queries: list[str], nb_chunks: int = 5
    ):
        contexts = []
        docs_name = []
        for query in queries:
            context, doc_name = self.get_rag_context(
                query=query, nb_chunks=nb_chunks
            )
            contexts.append(context)
            docs_name.append(docs_name)
        return contexts, docs_name

    def generate_answers(self, queries: list[str], nb_chunks: int = 2, options_generation = None):
        answers = []
        for query in queries:
            answer = self.generate_answer(query=query, 
                                          nb_chunks=nb_chunks,
                                          options_generation=options_generation)
            answers.append(answer)
        return answers
