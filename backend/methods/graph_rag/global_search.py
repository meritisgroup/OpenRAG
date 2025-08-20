from ...utils.agent import Agent

from ...base_classes import Search
from ...database.database_class import DataBase
from ...database.rag_classes import MergeEntityOverall, Community
from .prompts import PROMPTS

import json
import numpy as np



class GlobalSearch(Search):
    def __init__(
        self,
        agent: Agent,
        data_manager,
        pre_filter_size: int = 3,
        language: str = "EN",
    ) -> None:

        self.data_manager = data_manager
        self.pre_filter = pre_filter_size
        self.language = language
        self.agent = agent

    def get_context(self, query: str) -> str:
        """Generate the context from the query"""
        useful_communities = {}
        useful_entities = {}
        top_k_communities = self.data_manager.k_search(queries=query,
                                                       collection_name="global_search",
                                                       k=self.pre_filter)
        tokens_counter = {}
        tokens_counter["nb_input_tokens"] = 0
        tokens_counter["nb_output_tokens"] = 0

        for title in top_k_communities[0]:
                community = (
                    self.data_manager.query_filter(table_class=Community,
                                                   filter=(Community.title == title["text"]))
                    [0]
                )
                useful_communities[community.title] = community.description
                useful_entities[community.title] = community.entities_ids
        evaluated_communities = self._evaluate_communities(
            query=query,
            context=json.dumps(useful_communities, indent=4).replace('",', '",\n'),
        )
        tokens_counter["nb_input_tokens"] += np.sum(evaluated_communities["nb_input_tokens"])
        tokens_counter["nb_output_tokens"] += np.sum(evaluated_communities["nb_output_tokens"])

        try:
            dic_communities = json.loads(evaluated_communities["texts"])
        except Exception as e:
            print("Couldn't use evaluated communities for building the context :", e)
            dic_communities = useful_communities

        # dic_communities = useful_communities
        context = self._build_context(dic_communities, useful_entities)

        return context, tokens_counter

    def _build_context(self, dic_communities: dict, useful_entities: dict):
        context = "-- Context --\n"
        for community_title in dic_communities.keys():
            context += (
                f"Title : {community_title}\n{dic_communities[community_title]}\n\n"
            )

        entities = []
        for community_title in dic_communities.keys():

            if community_title in useful_entities.keys():

                entities += (
                    self.data_manager.query_filter(table_class=MergeEntityOverall, 
                                                   filter=MergeEntityOverall.id.in_(useful_entities[community_title]))
                )

        sorted_entities = sorted(entities, key=lambda x: int(x.degree), reverse=True)

        for entity in sorted_entities:
            context += f"{entity.name} : {entity.description}\n\n"

        return context

    def _evaluate_communities(self, query, context):

        system_prompt = PROMPTS[self.language]["COMMUNITY_EVALUATOR"]["SYSTEM_PROMPT"]
        prompt_template = PROMPTS[self.language]["COMMUNITY_EVALUATOR"]["QUERY_TEMPLATE"]

        context_base = dict(
            context=context,
            query=query,
        )
        prompt = prompt_template.format(**context_base)

        answer = self.agent.predict(system_prompt=system_prompt,
                                    prompt=prompt)
        return answer
