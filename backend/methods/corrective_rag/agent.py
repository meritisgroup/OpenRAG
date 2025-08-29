from ..naive_rag.agent import NaiveRagAgent
from .query import NaiveSearch
from .crawler import web_search
from ...utils.splitter import get_splitter
from tqdm import tqdm
import numpy as np
from .prompts import prompts
from ..query_reformulation.query_reformulation import query_reformulation


class CragAgent(NaiveRagAgent):

    def __init__(
        self,
        config_server: dict,
        dbs_name,
        data_folders_name,
        max_web_requests=2,
    ) -> None:

        super().__init__(
            config_server=config_server,
            dbs_name=dbs_name,
            data_folders_name=data_folders_name
        )
        self.language = config_server["language"]
        self.type_text_splitter = config_server["TextSplitter"]
        self.nb_chunks = config_server["nb_chunks"]
        self.max_web_requests = max_web_requests
        self.websearch = web_search(max_requests=self.max_web_requests)
        self.prompts = prompts[self.language]
        self.websplitter = get_splitter(
            type_text_splitter=self.type_text_splitter,
            agent=self.agent,
            embedding_model=self.embedding_model,
        )
        self.reformulate_query = config_server["reformulate_query"]
        if self.reformulate_query:
            self.reformulater = query_reformulation(
                agent=self.agent, language=self.language
            )

        self.prompts = prompts[self.language]


    def get_nb_token_embeddings(self):
        return self.data_manager.get_nb_token_embeddings()

    def get_rag_context(self, query: str, nb_chunks: int = 5) -> list[str]:
        """ """
        ns = NaiveSearch(data_manager=self.data_manager, nb_chunks=nb_chunks)
        context, docs_name = ns.get_context(query=query)
        return context, docs_name

    def web_results_refinement(
        self, web_results, query, model=None, batch: bool = True
    ):

        nb_input_tokens = 0
        nb_output_tokens = 0
        energies = [0, 0, ""]
        impacts = [0, 0, ""]
        useful_chunks = []
        if len(web_results) == 0:
            return useful_chunks, 0, 0
        for i in tqdm(range(len(web_results))):
            chunks = self.websplitter.split_text(text=web_results[i]["soup_html"])
            if batch:
                prompts = []
                system_prompts = []
                for j in range(len(chunks)):
                    prompt = self.prompts["document_relevance2"][
                        "QUERY_TEMPLATE"
                    ].format(context=chunks[j], query=query)

                    prompts.append(prompt)
                    system_prompts.append(self.system_prompt)
                scores = self.agent.multiple_predict(
                    prompts=prompts, system_prompts=system_prompts
                )
                nb_input_tokens += np.sum(scores["nb_input_tokens"])
                nb_output_tokens += np.sum(scores["nb_output_tokens"])
                energies[0] += scores["energy"][0]
                energies[1] += scores["energy"][1]

                impacts[0] += scores["impacts"][0]
                impacts[1] += scores["impacts"][1]
                scores = scores["texts"]
                for j in range(len(chunks)):
                    if (
                        "relevant" in scores[j].lower()
                        and "irrelevant" not in scores[j].lower()
                    ):
                        useful_chunks.append(chunks[j])

            else:
                nb_input_tokens = 0
                nb_output_tokens = 0
                for j in range(len(chunks)):
                    prompt = self.prompts["document_relevance2"][
                        "QUERY_TEMPLATE"
                    ].format(context=chunks[j], query=query)

                    score = self.agent.predict(
                        prompt=prompt, system_prompt=self.system_prompt
                    )
                    nb_input_tokens += np.sum(score["nb_input_tokens"])
                    nb_output_tokens += np.sum(score["nb_output_tokens"])
                    energies[0] += score["energy"][0]
                    energies[1] += score["energy"][1]

                    impacts[0] += score["impacts"][0]
                    impacts[1] += score["impacts"][1]
                    score = score["texts"]
                    if (
                        "relevant" in score.lower()
                        and "irrelevant" not in score.lower()
                    ):
                        useful_chunks.append(chunks[j])
        return {
            "texts": useful_chunks,
            "nb_input_tokens": nb_input_tokens,
            "nb_output_tokens": nb_output_tokens,
            "energy": energies,
            "impacts": impacts,
        }

    def web_search(self, query):
        try:
            results = self.websearch.search(query)
        except Exception:
            results = []
        return results

    def generate_answer(
        self,
        query: str,
        nb_chunks: str = 5,
        batch: bool = True,
        options_generation = None
    ) -> str:
        """Generate an answer to the query"""

        agent = self.agent

        nb_input_tokens = 0
        nb_output_tokens = 0
        impacts = [0, 0, ""]
        energies = [0, 0, ""]
        if self.reformulate_query:
            query, input_t, output_t, impacts, energies = self.reformulater.reformulate(
                query=query, nb_reformulation=1
            )
            query = query[0]
            nb_input_tokens += input_t
            nb_output_tokens += output_t
        context = ""
        contexts, docs_name = self.get_rag_context(query=query, nb_chunks=nb_chunks)
        useful_contexts = []
        ambiguous_contexts = []
        if batch:
            prompts = []
            system_prompts = []
            for j in range(len(contexts)):
                prompt = self.prompts["document_relevance2"]["QUERY_TEMPLATE"].format(
                    context=contexts[j], query=query
                )
                system_prompt = self.prompts["document_relevance2"]["SYSTEM_PROMPT"]
                prompts.append(prompt)
                system_prompts.append(system_prompt)
            scores = agent.multiple_predict(
                prompts=prompts, system_prompts=system_prompts
            )
            if "nb_input_tokens" in scores.keys():
                nb_input_tokens += np.sum(scores["nb_input_tokens"])
            if nb_output_tokens in scores.keys():
                nb_output_tokens += np.sum(scores["nb_output_tokens"])

            impacts[2] = scores["impacts"][2]
            impacts[0] += scores["impacts"][0]
            impacts[1] += scores["impacts"][1]

            energies[2] = scores["energy"][2]
            energies[0] += scores["energy"][0]
            energies[1] += scores["energy"][1]

            scores = scores["texts"]
            for j in range(len(contexts)):
                if (
                    "relevant" in scores[j].lower()
                    and "irrelevant" not in scores[j].lower()
                ):
                    useful_contexts.append(contexts[j])
                elif "ambiguous" in scores[j].lower():
                    ambiguous_contexts.append(contexts[j])
        else:
            for i, context in enumerate(contexts):
                prompt = self.prompts["document_relevance2"]["QUERY_TEMPLATE"].format(
                    context=contexts[i], query=query
                )
                system_prompt = self.prompts["document_relevance2"]["SYSTEM_PROMPT"]
                score = agent.predict(prompt=prompt, system_prompt=system_prompt)
                nb_input_tokens += scores["nb_input_tokens"]
                nb_output_tokens += scores["nb_output_tokens"]

                impacts[0] += score["impacts"][0]
                impacts[1] += score["impacts"][1]

                energies[0] += score["energy"][0]
                energies[1] += score["energy"][1]
                score = score["text"]
                if "relevant" in score.lower() and "irrelevant" not in score.lower():
                    useful_contexts.append(context)
                elif "ambiguous" in score.lower():
                    ambiguous_contexts.append(context)

        if len(useful_contexts) > 0:
            context = self.contexts_to_prompts(contexts=useful_contexts)
        elif len(ambiguous_contexts) > 0:
            prompt = self.prompts["rewrite_web_query"]["QUERY_TEMPLATE"].format(
                query=query
            )
            system_prompt = self.prompts["rewrite_web_query"]["SYSTEM_PROMPT"]

            answer = agent.predict(prompt=prompt, system_prompt=system_prompt)
            query_web = answer["texts"]
            nb_input_tokens += answer["nb_input_tokens"]
            nb_output_tokens += answer["nb_output_tokens"]

            impacts[0] += answer["impacts"][0]
            impacts[1] += answer["impacts"][1]

            energies[0] += answer["energy"][0]
            energies[1] += answer["energy"][1]

            web_results = self.web_search(query=query_web)
            if len(web_results) > 0:
                web_results = self.web_results_refinement(
                    web_results=web_results, query=query
                )
                nb_input_tokens += web_results["nb_input_tokens"]
                nb_output_tokens += web_results["nb_output_token"]
                energies[0] += web_results["energy"][0]
                energies[1] += web_results["energy"][1]
                impacts[0] += web_results["impacts"][0]
                impacts[1] += web_results["impacts"][1]

                web_results = web_results["texts"]
            ambiguous_contexts += web_results
            context = self.contexts_to_prompts(contexts=ambiguous_contexts)
        else:
            prompt = self.prompts["rewrite_web_query"]["QUERY_TEMPLATE"].format(
                query=query
            )
            system_prompt = self.prompts["rewrite_web_query"]["SYSTEM_PROMPT"]

            query_web = agent.predict(prompt=prompt, system_prompt=system_prompt)
            nb_input_tokens += (
                query_web["nb_input_tokens"]
                if type(query_web["nb_input_tokens"]) is type(0)
                else query_web["nb_input_tokens"][0]
            )
            nb_output_tokens += (
                query_web["nb_output_tokens"]
                if type(query_web["nb_output_tokens"]) is type(0)
                else query_web["nb_output_tokens"][0]
            )

            impacts[0] += query_web["impacts"][0]
            impacts[1] += query_web["impacts"][1]

            energies[0] += query_web["energy"][0]
            energies[1] += query_web["energy"][1]

            query_web = query_web["texts"]
            web_results = self.web_search(query=query_web)

            if len(web_results) > 0:
                web_results = self.web_results_refinement(
                    web_results=web_results, query=query
                )
                nb_input_tokens += web_results["nb_input_tokens"]
                nb_output_tokens += web_results["nb_output_tokens"]

                energies[0] += web_results["energy"][0]
                energies[1] += web_results["energy"][1]
                impacts[0] += web_results["impacts"][0]
                impacts[1] += web_results["impacts"][1]
                web_results = web_results["texts"]
                context = self.contexts_to_prompts(contexts=web_results)

        prompt = self.prompts["smooth_generation"]["QUERY_TEMPLATE"].format(
            context=context, query=query
        )


        if options_generation is None:
            options_generation = self.config_server["options_generation"]

        system_prompt = self.prompts["smooth_generation"]["SYSTEM_PROMPT"]
        answer = agent.predict(prompt=prompt, system_prompt=system_prompt,
                               options_generation=options_generation)
        nb_input_tokens += (
            answer["nb_input_tokens"]
            if type(answer["nb_input_tokens"]) is type(0)
            else answer["nb_input_tokens"][0]
        )
        nb_output_tokens += (
            answer["nb_output_tokens"]
            if type(answer["nb_output_tokens"]) is type(0)
            else answer["nb_output_tokens"][0]
        )

        impacts[0] += answer["impacts"][0]
        impacts[1] += answer["impacts"][1]

        energies[0] += answer["energy"][0]
        energies[1] += answer["energy"][1]

        return {
            "answer": answer["texts"],
            "nb_input_tokens": nb_input_tokens,
            "nb_output_tokens": nb_output_tokens,
            "context": context,
            "doc_names": [],
            "impacts": impacts,
            "energy": energies,
        }

    def release_gpu_memory(self):
        self.agent.release_memory()

    def get_rag_contexts(self, queries: list[str], nb_chunks: int = 5):
        contexts = []
        names_docs = []
        for query in queries:
            context, name_docs = self.get_rag_context(query=query, nb_chunks=nb_chunks)
            contexts.append(context)
            names_docs.append(name_docs)
        return contexts, names_docs

    def generate_answers(self, queries: list[str], nb_chunks: int = 2, options_generation = None):
        answers = []
        for query in queries:
            answer = self.generate_answer(query=query, nb_chunks=nb_chunks,
                                          options_generation=options_generation)
            answers.append(answer)
        return answers
