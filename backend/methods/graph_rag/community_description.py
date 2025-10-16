from ...utils.agent import Agent
from .graph_creation import Graph
from ...database.database_class import DataBase
from ...database.rag_classes import (
    MergeEntityDocument,
    MergeEntityOverall,
    Community,
    Entity,
    Relation,
    Tokens,
    CommunityEntity,
    CommunityRelation,
)
from .prompts import PROMPTS
import json
import re
import numpy as np


class CommunityDescription:
    """
    This is the class built to describe communities of a graph and save it in a database
    """

    def __init__(self, agent: Agent, model: str, graph: Graph, db) -> None:
        """ """
        if graph.communities == []:
            graph.create_communities()

        self.graph = graph
        self.agent = agent
        self.llm_model = model
        self.db = db

    def get_context_for_community(
        self, ids: list[int], overall=True, language="EN", community_as_entity=False
    ) -> tuple[str, str]:
        """
        From a list of int representing entities' ids, the prompt and the system prompt to generate the community description
        """
        system_prompt = PROMPTS[language]["summarize_community"]["SYSTEM_PROMPT"]
        query_prompt_template = PROMPTS[language]["summarize_community"]["QUERY_TEMPLATE"]

        if overall:
            entities: list[Entity] = (
                self.db.query_filter(table_class=MergeEntityOverall, 
                                               filter=MergeEntityOverall.id.in_(ids))
            )
            doc_name = "all"
            entities_names = [entity.name for entity in entities]
            relations = (
                self.db.query_filter(table_class=Relation,
                                                filter=(Relation.source.in_(entities_names) & Relation.target.in_(entities_names)))

            )

        elif not overall and not community_as_entity:
            entities = (
                self.db.query_filter(table_class=MergeEntityDocument,
                                               filter=MergeEntityDocument.id.in_(ids))
                    )
            doc_name = entities[0].doc_name
            entities_names = [entity.name for entity in entities]
            relations = (
                self.db.query_filter(table_class=Relation,
                                               filter=(Relation.source.in_(entities_names)
                                                       & Relation.target.in_(entities_names)
                                                       & Relation.doc_name
                                                        == doc_name)
                                              )
                        )       

        else:
            entities = (
                self.db.query_filter(table_class=CommunityEntity,
                                               filter=CommunityEntity.id.in_(ids)))
            doc_name = "all"
            entities_names = [entity.name for entity in entities]
            relations = (
                self.db.query_filter(table_class=CommunityRelation, 
                                               filter=(CommunityRelation.source.in_(entities_names)
                                                       & CommunityRelation.target.in_(entities_names)
                                                      ))
                        )

        sorted_entities = sorted(entities,
                                 key=lambda x: x.degree,
                                 reverse=True)

        context = "Here are provided entities you have to create a consistent title and a consistent description :\n\n"
        for entity in sorted_entities:
            context += f"\t- {entity.name} (retrieved {entity.degree} times in text) : {entity.description}\n\n"

        context += "\nHere are relationships between those entities, retrieved in documents :\n\n"
        for relation in relations:
            context += f"\t- ({relation.source}, {relation.target}) : {relation.description}\n\n"

        context_base = dict(
            language=language,
            list_descriptions=context,
        )
        prompt = query_prompt_template.format(**context_base)

        return prompt, system_prompt

    def process_communities(
        self,
        overall=True,
        language="EN",
        deep_level=1,
        community_as_entity=False,
    ):

        prompts_to_process = []
        system_prompts_to_process = []
        communities_to_process = []
        for community in self.graph.communities:

            already_treated = (
                len(
                    self.db.query_filter(table_class=Community, 
                                                   filter=Community.entities_ids == community)
                )
                > 0
            )

            if not already_treated:

                prompt, system_prompt = self.get_context_for_community(
                    ids=community,
                    overall=overall,
                    language=language,
                    community_as_entity=community_as_entity,
                )
                communities_to_process.append(community)
                prompts_to_process.append(prompt)
                system_prompts_to_process.append(system_prompt)

        tokens = 0
        taille_batch = 100
        outputs = None
        for i in range(0, len(prompts_to_process), taille_batch):
            results = self.agent.multiple_predict(prompts=prompts_to_process[i:i + taille_batch],
                                                  system_prompts=system_prompts_to_process[i:i + taille_batch],
                                                  model=self.llm_model)
            if outputs is None:
                outputs = results
            else:
                outputs["texts"].append(results["texts"])
                outputs["nb_input_tokens"]+=np.sum(results["nb_input_tokens"])
                outputs["nb_output_tokens"]+=np.sum(results["nb_output_tokens"])
                outputs["impacts"][0]+=results["impacts"][0]
                outputs["impacts"][1]+=results["impacts"][1]
                outputs["energy"][0]+=results["energy"][0]
                outputs["energy"][1]+=results["energy"][1]
        llm_outputs = outputs["texts"]
        input_tokens = np.sum(outputs["nb_input_tokens"])
        output_tokens = np.sum(outputs["nb_output_tokens"])

        new_communities = self.clean_outputs(
            llm_outputs, communities_to_process, deep_level
        )

        for new_community in new_communities:
            self.db.add_instance(new_community)

        communities_tokens = Tokens(
            title="communities",
            embedding_tokens=0,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
        )
        self.db.add_instance(communities_tokens)

        print("Processing communities - ✅")

    def _handle_single_community(self, output):
        try:
            data = json.loads(output)
            title = data.get("title", "")
            description = data.get("description", "")

            return title, description
        except json.JSONDecodeError:
            title_match = re.search(r'"title"\s*:\s*"([^"]*)"', output)
            description_match = re.search(r'"description"\s*:\s*"([^"]*)"', output)

            title = title_match.group(1) if title_match else None
            description = description_match.group(1) if description_match else None

            return title, description

    def clean_outputs(
        self, outputs: list[str], communities_to_process, deep_level
    ) -> list[Community]:

        new_communities = []
        for output, community in zip(outputs, communities_to_process):
            title, description = self._handle_single_community(output)
            if title and description is not None:
                new_community = Community(
                    title=title,
                    description=description,
                    entities_ids=community,
                    hierarchical_level=deep_level,
                )
                new_communities.append(new_community)

        return new_communities
    
    def get_database(self):
        return self.db
