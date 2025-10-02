# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:37:47 2025

@author: chardy
"""
from ..naive_rag.agent import NaiveRagAgent
from ...utils.agent import get_Agent
from .prompts import prompts
import numpy as np
from .get_rags_to_merge import get_rag_to_merge


class MergerRagAgent(NaiveRagAgent):
    "Original RAG with no modification"

    def __init__(
        self,
        config_server: dict,
        dbs_name: list[str],
        data_folders_name: list[str]
    ) -> None:
        
        """
        Args:

        """
        self.config_server=config_server
        self.rag_list=config_server["rag_list"]
        self.agent = get_Agent(config_server)
        self.nb_chunks = config_server["nb_chunks"]
        self.dbs_name = dbs_name
        self.data_folders_name = data_folders_name

        self.rag_agents = {}
        for i in range(len(self.rag_list)):
            rag_agent = get_rag_to_merge(rag_name=self.rag_list[i], 
                                         config_server=self.config_server["rag_config_list"][i],
                                         databases_name=data_folders_name)
            self.rag_agents[self.rag_list[i]] = rag_agent


    def get_infos_embeddings(self):
        infos = {}
        for i in range(len(self.rag_list)):
            temp_infos = self.rag_agents[self.rag_list[i]].get_infos_embeddings()
            for key in temp_infos.keys():
                if key not in infos.keys():
                    infos[key] = temp_infos[key]
                else:
                    infos[key]+=temp_infos[key]
        return infos

    def indexation_phase(
        self,
        reset_index: bool = False,
        overlap: bool = True,
        reset_preprocess = False
    ) -> None:
        """
        Does the indexation of a given knowledge base, full process is located in indexation.py
        Args:
            path_input (str) : where the documents to be processed are stored
            chunk_size (str) : number of characters in each chunk
            overlap (bool) : Wether chunks overlap each other

        """
        if reset_preprocess:
            reset_index = True
            
        if reset_index:
            for rag_name in self.rag_list:
                self.rag_agents[rag_name].data_manager.delete_collection()
                self.rag_agents[rag_name].data_manager.clean_database()

        for rag_name in self.rag_list:
            self.rag_agents[rag_name].indexation_phase(reset_index = reset_index,
                                                       overlap = overlap,
                                                       reset_preprocess = reset_preprocess)
            if reset_preprocess:
                reset_preprocess = False
        


    def merge_answers(self, answers: list[str], query: str, agent, options_generation=None) -> str:

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

        return agent.predict(prompt=user_prompt,
                             system_prompt=system_prompt,
                             options_generation=options_generation)

    def generate_answer(
        self,
        query: str,
        nb_chunks: int = 5,
        options_generation = None
    ) -> str:
       
        self.list_answers={}
        full_response={
                "answer": "",
                "nb_input_tokens": 0,
                "nb_output_tokens": 0,
                "context": [],
                "impacts": [0,0,""],
                "energy": [0,0,""],
            }

        for rag_name in self.rag_list:
            response=self.rag_agents[rag_name].generate_answer(query=query,
                                                               nb_chunks=nb_chunks,
                                                               options_generation=options_generation)
            self.list_answers[rag_name] = response["answer"]
            full_response["nb_input_tokens"]+=response["nb_input_tokens"]
            full_response["nb_output_tokens"]+=response["nb_output_tokens"]
            full_response["context"] += response["context"]
            full_response["impacts"][0]+=response["impacts"][0]
            full_response["impacts"][1]+=response["impacts"][1]
            full_response["impacts"][2]+=response["impacts"][2]
            full_response["energy"][0]+=response["energy"][0]
            full_response["energy"][1]+=response["energy"][1]
            full_response["energy"][2]+=response["energy"][2]            
            

        merged_answer = self.merge_answers(answers=list(self.list_answers.values()),
                                            query=query,
                                            agent=self.agent,
                                            options_generation=options_generation
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
                "original_query": query
            }
