from ...utils.agent import Agent
from ...database.database_class import DataBase
from ...database.rag_classes import MergeEntityOverall, Relation, Chunk
from .prompts import PROMPTS

import ast
import re


class LocalSearch:
    def __init__(
        self,
        agent: Agent,
        data_manager,
        start_node: int = 5,
        language: str = "EN",
    ) -> None:

        self.data_manager
        self.start_node = start_node
        self.language = language
        self.agent = agent

        self.tokens_counter = {}

    def retrieve_from_a_query(self, query: str) -> list[str]:
        """
        Returns the list of entities retrieved in the query
        """

        system_prompt, prompt = self._get_extraction_query_prompt(query)

        answer = self.agent.predict(prompt=prompt,
                                    system_prompt=system_prompt)

        self.tokens_counter["nb_input_tokens"] = answer["nb_input_tokens"]
        self.tokens_counter["nb_output_tokens"] = answer["nb_output_tokens"]

        match = re.search(r"\[.*?\]", answer["texts"])

        if match:
            entities_found = match.group(0)
            entities_list = ast.literal_eval(entities_found)
            return entities_list
        else:
            return None

    def _get_extraction_query_prompt(self, query: str) -> list[str] | None:

        system_prompt = PROMPTS[self.language]["extration_query"]["SYSTEM_PROMPT"]
        prompt_template = PROMPTS[self.language]["extration_query"]["QUERY_TEMPLATE"]

        context_base = dict(
            query=query,
        )

        prompt = prompt_template.format(**context_base)

        return system_prompt, prompt

    def get_context(self, query: str) -> str:
        """
        Returns the description of entities found in the query and their relations
        """

        self.tokens_counter["nb_input_tokens"] = 0
        self.tokens_counter["nb_output_tokens"] = 0

        # First we extract entitis from the query
        entities_in_query = self.retrieve_from_a_query(query)

        if entities_in_query == [] or entities_in_query is None:
            # print("No available context found for this query")
            return "No available context found for this query.\nPlease juste use your knowledge to answer."

        # We associate each entity extracted with self.start_node entities in the data base
        entities, entities_names = [], []

        for entity in entities_in_query:

            if type(entity) is type("str") and entity is not None:
                closest_entities = self.data_manager.k_search(
                    queries=entity,
                    collection_name="local_search",
                    k=self.start_node,
                    output_fields=["text"],
                )

                for closest_entity in closest_entities[0]:
                    entity_in_db = (
                        self.data_manager.query_filter(table_class=MergeEntityOverall,
                                                      filter=MergeEntityOverall.name == closest_entity["text"])[0]
                    )

                    if entity_in_db not in entities:
                        entities.append(entity_in_db)
                        entities_names.append(entity_in_db.name)

        # For each entity, we retrieve its relations
        both_relations, single_relations = [], []

        for relation in self.data_manager.query(Relation):
            if relation.source in entities_names and relation.target in entities_names:
                both_relations.append(relation)

            elif relation.source in entities_names or relation.target in entities_names:
                single_relations.append(relation)

            else:
                pass

        context, chunks = self._build_context(entities, both_relations, single_relations)

        return context, chunks, self.tokens_counter

    def _build_context(self, entities, both_relations, single_relations) -> str:

        context_template = PROMPTS[self.language]["local_search_contxt_template"]
        adding_entities = ""
        adding_relations = ""

        adding_entity = PROMPTS[self.language]["unique_entity"]
        adding_relation = PROMPTS[self.language]["unique_relation"]

        chunks = []
        for entity in entities:
            adding_entities += (
                adding_entity.replace("{entity}", entity.name).replace(
                    "{entity_description}", entity.description.replace("\n", "")
                )
                + "\n\n"
            )
            new_chunk = Chunk(text=adding_entity.replace("{entity}", entity.name).replace("{entity_description}",
                                                                                                 entity.description.replace("\n", "")),
                                                id=i)
            chunks.append(new_chunk)
            i+=1

        for relation in both_relations:
            adding_relations += (
                adding_relation.replace("{source}", relation.source).replace("{target}", relation.target).replace("{relation_description}", relation.description)
                + "\n\n"
            )
            new_chunk = Chunk(text=adding_relation.replace("{source}",
                                                                  relation.source).replace("{target}",
                                                                                            relation.target).replace("{relation_description}",
                                                                                                                      relation.description),
                                    id=i)
            chunks.append(new_chunk)
            i+=1

        for relation in single_relations:
            adding_relations += (
                adding_relation.replace("{entity_source}", relation.source)
                .replace("{entity_target}", relation.target)
                .replace("{relation_description}", relation.description)
                + "\n\n"
            )
            new_chunk = Chunk(text=adding_relation.replace("{entity_source}",
                                                                  relation.source).replace("{entity_target}",
                                                                                            relation.target).replace("{relation_description}", 
                                                                                                                     relation.description),
                                    id=i)
            chunks.append(new_chunk)
            i+=1
        context = context_template.replace("{entities}", adding_entities).replace(
            "{relations}", adding_relations
        )

        return context, chunks
