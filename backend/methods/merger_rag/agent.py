# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:37:47 2025

@author: chardy
"""
from ...base_classes import RagAgent
from ...utils.agent import get_Agent
from .prompts import prompts
import numpy as np
from .get_rags_to_merge import get_rag_to_merge


class MergerRagAgent(RagAgent):
    "Original RAG with no modification"

    def __init__(
        self,
        config_server: dict,
    ) -> None:
        
        """
        Args:

        """
        self.config_server=config_server
        self.rag_list=config_server["rag_list"]
        self.agent = get_Agent(config_server)
        self.nb_chunks = config_server["nb_chunks"]
   

    def indexation_phase(
        self, path_input : str, chunk_size : int=500, overlap : bool = True, reset_index: bool=False 
    ) -> None:
        """
        Does the indexation of a given knowledge base, full process is located in indexation.py
        Args:
            path_input (str) : where the documents to be processed are stored
            chunk_size (str) : number of characters in each chunk
            overlap (bool) : Wether chunks overlap each other

        """
        
        self.rag_agents = {}
        for rag_name in self.rag_list:

            
            rag_agent = get_rag_to_merge(rag_name, self.config_server, database_name="")
            self.rag_agents[rag_name] = rag_agent
            if hasattr(rag_agent,"indexation_phase"):
                self.rag_agents[rag_name].indexation_phase(
            path_input = path_input,
            reset_index = reset_index,
            chunk_size = chunk_size,
            overlap = overlap,
        )
            


        if reset_index:
            for rag_name in self.rag_list:
                if rag_name!="naive_chatbot":
                    self.rag_agents[rag_name].vb.delete_collection()
                    self.rag_agents[rag_name].db.clean_database()
        



    def merge_answers(self, answers: list[str], query: str, agent) -> str:

        

        formatted_answers = "\n\n".join(
            [f"Réponse {i+1} :\n{ans}" for i, ans in enumerate(answers)]
        )

        system_prompt = (
            "Tu es un assistant expert en synthèse d'information. "
            "Tu dois combiner plusieurs réponses à une même question pour produire une réponse unique et complète"
        )

        user_prompt = (
            f"Voici la question :\n{query}\n\n"
            f"Tu as reçu {len(answers)} réponses de différents systèmes RAG. "
            f"Voici ces réponses :\n\n{formatted_answers}\n\n"
            f"Ta tâche est de produire une réponse synthétique, précise et complète. "
            f"Si plusieurs réponses disent la même chose, ne le répète pas inutilement. "
            f"S'il y a des divergences, essaie de les expliquer ou de les concilier."
            
        )

        return agent.predict(prompt=user_prompt, system_prompt=system_prompt)

  

    def generate_answer(
        self,
        query: str,
        nb_chunks: int = 5,
    ) -> str:
       
        self.list_answers={}
        full_response={
                "answer": "",
                "nb_input_tokens": 0,
                "nb_output_tokens": 0,
                "context": "",
                "impacts": [0,0,""],
                "energy": [0,0,""],
            }

        for rag_name in self.rag_list:
            response=self.rag_agents[rag_name].generate_answer(
        query=query,
        nb_chunks=nb_chunks)
            self.list_answers[rag_name] = response["answer"]

            
            full_response["nb_input_tokens"]+=response["nb_input_tokens"]
            full_response["nb_output_tokens"]+=response["nb_output_tokens"]
            full_response["context"]+=response["context"]
            full_response["impacts"][0]+=response["impacts"][0]
            full_response["impacts"][1]+=response["impacts"][1]
            full_response["impacts"][2]+=response["impacts"][2]
            full_response["energy"][0]+=response["energy"][0]
            full_response["energy"][1]+=response["energy"][1]
            full_response["energy"][2]+=response["energy"][2]
            
            

        merged_answer = self.merge_answers(answers=list(self.list_answers.values()),
        query=query,
        agent=self.agent
    )
        
        full_response["answer"]=merged_answer['texts']


        full_response["nb_input_tokens"]+=np.sum(merged_answer["nb_input_tokens"])
        full_response["nb_output_tokens"]+=np.sum(merged_answer["nb_output_tokens"])
        full_response["impacts"][0]+=merged_answer["impacts"][0]
        full_response["impacts"][1]+=merged_answer["impacts"][1]
        full_response["impacts"][2]+=merged_answer["impacts"][2]
        full_response["energy"][0]+=merged_answer["energy"][0]
        full_response["energy"][1]+=merged_answer["energy"][1]
        full_response["energy"][2]+=merged_answer["energy"][2]

        
        return {
                "answer": full_response["answer"],
                "nb_input_tokens": full_response["nb_input_tokens"],
                "nb_output_tokens": full_response["nb_output_tokens"],
                "context": full_response["context"],
                "impacts": full_response["impacts"],
                "energy": full_response["energy"],
            }
    

    def release_gpu_memory(self):
        self.agent.release_memory()

    

    def generate_answers(self, queries: list[str], nb_chunks: int = 2):
        answers = []
        for query in queries:
            answer = self.generate_answer(query=query, nb_chunks=nb_chunks)
            answers.append(answer)
        return answers
