from utils.agent import Agent

class DescriptionClean:

    def __init__(self, descriptions: list[str]):
        self.descriptions = descriptions

    def clean_description(self, agent: Agent) -> str:
        final_merged_description = '\n'.join((description for description in self.descriptions))
        return final_merged_description