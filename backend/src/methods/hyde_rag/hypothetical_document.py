from .prompts import prompts

class HypotheticalDocumentGenerator:

    def __init__(self, agent, model, language):
        self.agent = agent
        self.model = model
        self.language = language
        self.prompts = prompts[language]

    def generate_hypothetical_document(self, query):
        """
        Generate a hypothetical document that would answer the user query.
        This document will be used for embedding-based retrieval instead of the query itself.
        """
        prompt = self.prompts['hypothetical_document']['QUERY_TEMPLATE'].format(query=query)
        system_prompt = self.prompts['hypothetical_document']['SYSTEM_PROMPT']

        response = self.agent.predict(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.model
        )

        hypothetical_doc = response['texts']
        nb_input_tokens = response['nb_input_tokens']
        nb_output_tokens = response['nb_output_tokens']
        impacts = response['impacts']
        energy = response['energy']

        return (hypothetical_doc, nb_input_tokens, nb_output_tokens, impacts, energy)
