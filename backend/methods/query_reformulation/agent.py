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
        db_name: str = "db_naive_rag",
        vb_name: str = "vb_naive_rag",
    ) -> None:

        super().__init__(
            config_server=config_server,
            db_name=db_name,
            vb_name=vb_name,
        )
        self.nb_chunks = config_server["nb_chunks"]
        self.prompts = prompts[self.language]
        self.reformulater = query_reformulation(
            agent=self.agent, language=self.language
        )

    def get_nb_token_embeddings(self):
        return self.vb.get_nb_token_embeddings()

    def get_rag_context(
        self, query: str, nb_chunks: int = 5
    ) -> list[str]:
        ns = NaiveSearch(vector_base=self.vb, nb_chunks=nb_chunks)
        context = ns.get_context(query=query)
        return context

    def contexts_to_prompts(self, contexts):
        context = ""
        for chunk in contexts:
            if chunk not in context:
                context += chunk + "\n[...]\n"
        return context

    def generate_answer(
        self,
        query: str,
        nb_chunks: int = 5,
        nb_reformulation: int = 5
    ) -> str:
        """Generate an answer to the query"""
        agent = self.agent

        queries, nb_input_tokens, nb_output_tokens, impact, energy = self.reformulater.reformulate(
            query=query, nb_reformulation=nb_reformulation
        )
        contexts = [
            self.get_rag_context(query=query, nb_chunks=nb_chunks)
            for query in queries
        ]
        contexts = list(chain(*contexts))
        contexts = list(set(contexts))
        context = self.contexts_to_prompts(contexts=contexts)
        prompt = self.prompts["smooth_generation"]["QUERY_TEMPLATE"].format(
            context=context, query=query
        )
        system_prompt = self.prompts["smooth_generation"]["SYSTEM_PROMPT"]
        answer = agent.predict(prompt=prompt, system_prompt=system_prompt)
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
            "impacts": impacts,
            "energy" : energies
        }

    def release_gpu_memory(self):
        self.agent.release_memory()

    def get_rag_contexts(
        self, queries: list[str], nb_chunks: int = 5
    ):
        contexts = []
        names_docs = []
        for query in queries:
            context, name_docs = self.get_rag_context(
                query=query, nb_chunks=nb_chunks
            )
            contexts.append(context)
            names_docs.append(name_docs)
        return contexts, names_docs

    def generate_answers(self, queries: list[str], nb_chunks: int = 2):
        answers = []
        for query in queries:
            answer = self.generate_answer(query=query, nb_chunks=nb_chunks)
            answers.append(answer)
        return answers
