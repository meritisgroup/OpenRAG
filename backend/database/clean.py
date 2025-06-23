from ..utils.agent import Agent


class DescriptionClean:
    def __init__(self, descriptions: list[str]):
        self.descriptions = descriptions

    def clean_description(self, agent: Agent) -> str:

        # if len(self.descriptions) == 1:
        #     return self.descriptions[0]

        # remaining_descriptions = []

        # for k in range(0, len(self.descriptions), 10):

        #     filtered_descriptions = agent.predict(
        #         query=str(self.descriptions[k : min(k + 10, len(self.descriptions))]),
        #         task_type="filter_descriptions",
        #     )["texts"]

        #     merged_description = agent.predict(
        #         query=str(filtered_descriptions), task_type="merge_descriptions"
        #     )["texts"]

        #     remaining_descriptions.append(merged_description)

        # if len(remaining_descriptions) == 1:
        #     return remaining_descriptions[0]

        # else:
        #     final_merged_description = agent.predict(
        #         query=str(remaining_descriptions), task_type="merge_descriptions"
        #     )["texts"]
        #     return final_merged_description
        final_merged_description = "\n".join(
            description for description in self.descriptions
        )
        return final_merged_description
