# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:37:47 2025

@author: chardy
"""
from .indexation import AgenticRagIndexation
from .query import NaiveSearch
from ...base_classes import RagAgent
from ...utils.factory_vectorbase import get_vectorbase
from ...database.database_class import get_database
from ...utils.agent import get_Agent
from .prompts import prompts
import numpy as np
from pydantic import BaseModel
from ..query_reformulation.query_reformulation import query_reformulation


class CompareQueryAnswer(BaseModel):
    Decision: bool

class AgenticRagAgent(RagAgent):
    "Original RAG with no modification"

    def __init__(
        self,
        config_server: dict,
        db_name: str = "db_agentic_rag",
        vb_name: str = "vb_agentic_rag"
    ) -> None:
        """
        Args:
            model (str): model used to generate answer, to be set in backend/methods/naive_rag/config.json file
            storage_path: folder in which database will be stored
            params_host_llm(dict): parameters for Ollama or VLLM, to be set in backend/config_server.json file
            params_vectorbase(dict): vectorbase connection parameters, to be set in backend/config_server.json file
            embedding_model (str): Model used to embed documents and queries, to be set in backend/methods/naive_rag/config.json file
            language (str) : Sets the language of the prompts (available "EN", "FR"), to be set in in backend/methods/naive_rag/config.json file
            api_key (str) : API key to be used if needed, to be set in backend/config_server.json file (not mandatory if using Ollama or VLLM)
            db_name (str) : Name given to the database that keeps track of already processed docs, if it already exists adds new documents to the existing database (stored in storage/ folder)
            vb_name (str) : Name given to the vectorbase, if it already exists adds new documents to the existing vectorbase (stored in milvus/elasticsearch docker)
            type_retrieval (str) : How documents will be retrieved (embeddings, BM25, vlm_embeddings are available plus hybrid if using elasticsearch)
        """

        self.storage_path = config_server["storage_path"]
        self.nb_chunks = config_server["nb_chunks"]
        self.embedding_model = config_server["embedding_model"]
        self.language = config_server["language"]
        self.type_text_splitter = config_server["TextSplitter"]
        self.type_retrieval = config_server["type_retrieval"]

        self.db_name = db_name
        self.vb_name = vb_name

        self.params_vectorbase = config_server["params_vectorbase"]
        self.db = get_database(db_name=self.db_name, storage_path=self.storage_path)

        self.agent = get_Agent(config_server)
        self.config_server = config_server

        self.vb = get_vectorbase(
            vb_name=self.vb_name, config_server=config_server, agent=self.agent
        )
        self.prompts = prompts[self.language]
        self.reformulate_query = config_server["reformulate_query"]
        if self.reformulate_query:
            self.reformulater = query_reformulation(
                agent=self.agent, language=self.language
            )

    def get_nb_token_embeddings(self):
        return self.vb.get_nb_token_embeddings()

    def indexation_phase(
        self,
        path_input: str,
        reset_index: bool = False,
        chunk_size: int = 500,
        overlap: bool = True,
    ) -> None:
        """
        Does the indexation of a given knowledge base, full process is located in indexation.py
        Args:
            path_input (str) : where the documents to be processed are stored
            chunk_size (str) : number of characters in each chunk
            overlap (bool) : Wether chunks overlap each other

        """

        if reset_index:
            self.vb.delete_collection()
            self.db.clean_database()

        index = AgenticRagIndexation(
            data_path=path_input,
            db=self.db,
            vb=self.vb,
            type_text_splitter=self.type_text_splitter,
            agent=self.agent,
            embedding_model=self.embedding_model,
        )

        index.run_pipeline(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            batch=self.params_vectorbase["batch"],
        )

        return None

    def get_rag_context(
        self, query: str, nb_chunks: int = 5
    ) -> list[str]:
        """
        Takes a query and retrieves a given number of chunks using the NaiveSearch implemented

        Args:
            query (str) : The query that needs answering
            nb_chunks (int) : Number of chunks to be retrieved
        Output:
            context (list[str]) : All retrieved chunks
        """
        ns = NaiveSearch(vector_base=self.vb,
                         nb_chunks=nb_chunks)
        context, docs_name = ns.get_context(query=query)
        return context, docs_name


    def evaluate(self, query: str, answer: str, agent) -> bool:
        """
        Uses an LLM to determine whether the given answer fully satisfies the original question.

        Args:
            query (str): The user's original question.
            answer (str): The generated answer so far.
            agent (Agent): An Agent object with a `predict` method.

        Returns:
            bool: True if the answer is judged to be complete and correct, False otherwise.
        """
        system_prompt = (
            "You are an expert assistant tasked with evaluating whether an answer is complete with respect to a question.\n"
            "You must respond only with 'True' if the answer is complete and correct, otherwise 'False'."
        )

        user_prompt = f"""
        Question:
        {query}

        Answer:
        {answer}

        Does the answer fully and satisfactorily address the question? Reply only with 'True' or 'False'.
        """

        result = agent.predict_json(system_prompt=system_prompt,
                                    prompt=user_prompt,
                                    json_format=CompareQueryAnswer)
        

        return result.Decision

    

    def reformulate(self, query: str, answer: str, agent) -> str:
        """
        Compares the original query and the generated answer, then generates a new query
        that targets only the missing information in the answer.

        Args:
            query (str): The user's original question.
            answer (str): The previously generated answer.
            agent: The LLM agent used for query reformulation.

        Returns:
            str: A new query focused solely on the missing elements.
        """
        
        system_prompt = (
            "You are an AI assistant tasked with improving a generated answer.\n"
            "You are given an original user question and an answer previously produced by another model.\n"
            "You must analyze what is missing from the answer in relation to the question.\n"
            "Then generate ONE NEW QUERY that targets ONLY the missing information, "
            "and that can be used to complete the original answer.\n"
            "Your new query must be directly reusable by an LLM to produce a complementary answer.\n"
            "Do NOT rephrase the entire question. Do NOT repeat what has already been answered. "
            "Focus ONLY on the missing parts.\n"
        )

        user_prompt = (
            f"--- Original question ---\n{query}\n\n"
            f"--- Generated answer ---\n{answer}\n\n"
            "What new query would you ask to retrieve only the missing information?"
        )

        new_query = agent.predict(system_prompt=system_prompt, prompt=user_prompt)

        return new_query
    

    def concatene(self, answer_init: str, answer_add: str, query: str, agent) -> str:
        """
        Combines the initial and additional answers into a single, complete response
        that best answers the original user query.

        Args:
            answer_init (str): The initial answer generated by the system.
            answer_add (str): The additional answer obtained from a follow-up query.
            query (str): The original user question.
            agent: The LLM agent used to synthesize the final answer.

        Returns:
            str: A single, coherent, and complete response answering the original query.
        """

        system_prompt = (
            "You are an AI assistant. Your task is to synthesize two partial answers "
            "into a single, well-written and complete response to a given user question.\n"
            "Do not repeat content unnecessarily. Combine both answers intelligently and logically.\n"
            "The final answer must be clear, comprehensive, and directly address the user's query."
        )

        user_prompt = f"""
    --- User Question ---
    {query}

    --- Initial Answer ---
    {answer_init}

    --- Additional Answer ---
    {answer_add}

    --- Task ---
    Combine the two answers above into a single complete response that fully addresses the user's question.
    """

        final_answer = agent.predict(system_prompt=system_prompt, prompt=user_prompt)

        return final_answer['texts']




    def generate_intermediate_answer(
        self,
        query: str,
        nb_chunks: int = 5,
    ) -> str:
        """
        Takes a query, retrieves appropriated context and generates an answer
        Args:
            query (str) : The query that needs answering
            model (str) : name of the model used to answer
            nb_chunks (int) : number of chunks to retrieve

        Output:
            answer (str) : The answer to the query given the retrieved context
        """
        agent = self.agent
        nb_input_tokens = 0
        nb_output_tokens = 0
        impacts, energies = [0, 0, ""], [0, 0, ""]
        if self.reformulate_query:
            query, input_t, output_t, impacts, energies = self.reformulater.reformulate(
                query=query, nb_reformulation=1
            )
            query = query[0]
            nb_input_tokens += input_t
            nb_output_tokens = output_t

        context, _ = self.get_rag_context(
            query=query, nb_chunks=nb_chunks
        )
        prompt = self.prompts["smooth_generation"]["QUERY_TEMPLATE"].format(
            context=context, query=query
        )
        system_prompt = self.prompts["smooth_generation"]["SYSTEM_PROMPT"]
        answer = agent.predict(prompt=prompt, system_prompt=system_prompt)
        nb_input_tokens += np.sum(answer["nb_input_tokens"])
        nb_output_tokens += np.sum(answer["nb_output_tokens"])
        impacts[2] = answer["impacts"][2]
        impacts[0] += answer["impacts"][0]
        impacts[1] += answer["impacts"][1]
        energies[2] = answer["energy"][2]
        energies[0] += answer["energy"][0]
        energies[1] += answer["energy"][1]

        return {
            "answer": answer["texts"],
            "nb_input_tokens": nb_input_tokens,
            "nb_output_tokens": nb_output_tokens,
            "context": context,
            "impacts": impacts,
            "energy": energies,
        }


    
    def generate_answer(
        self,
        query: str,
        nb_chunks: int = 5, max_iter=0
    ) -> str:
        """
        Takes a query, retrieves appropriated context and generates an answer
        Args:
            query (str) : The query that needs answering
            model (str) : name of the model used to answer
            nb_chunks (int) : number of chunks to retrieve

        Output:
            answer (str) : The answer to the query given the retrieved context
        """
        agent = self.agent
        

        iter = 0
        info = self.generate_intermediate_answer(query, nb_chunks=5)
        answer = info["answer"]
        

        nb_input_tokens = info["nb_input_tokens"]
        nb_output_tokens = info["nb_output_tokens"]
        context_tot = info["context"]
        impacts, energies = info["impacts"], info["energy"]

        print(f"[DEBUG] Is initial answer sufficient? {self.evaluate(query, answer, agent)}")

        while iter <= max_iter and not self.evaluate(query, answer, agent):
            iter += 1
            print(f"[DEBUG] Iteration {iter} started")

            query_additional = self.reformulate(query, answer, agent)['texts']
            print(f"[DEBUG] Reformulated query for missing info: {query_additional}")

            info = self.generate_intermediate_answer(query_additional, nb_chunks=5)
            answer_additional = info["answer"]
            

            nb_input_tokens += info["nb_input_tokens"]
            nb_output_tokens += info["nb_output_tokens"]
            context_tot += info["context"]

            impacts[0] += info["impacts"][0]
            impacts[1] += info["impacts"][1] 
            impacts[2] = info["impacts"][2]

            energies[0] += info["energy"][0]
            energies[1] += info["energy"][1]
            energies[2] = info["energy"][2]

            answer = self.concatene(answer, answer_additional, query, agent)
            
            print(f"[DEBUG] Is updated answer sufficient? {self.evaluate(query, answer, agent)}")


        return {
            "answer": answer,
            "nb_input_tokens": nb_input_tokens,
            "nb_output_tokens": nb_output_tokens,
            "context": context_tot,
            "impacts": impacts,
            "energy": energies,
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
                query=query, nb_chunks=nb_chunks)
            contexts.append(context)
            names_docs.append(name_docs)
        return contexts, names_docs

    def generate_answers(self, queries: list[str], nb_chunks: int = 2):
        answers = []
        for query in queries:
            answer = self.generate_answer(query=query, nb_chunks=nb_chunks)
            answers.append(answer)
        return answers
