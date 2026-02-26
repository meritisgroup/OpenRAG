from .prompts import prompts

def remove_duplicates(lst):
    return list(dict.fromkeys(lst))

class query_reformulation:

    def __init__(self, agent, model, language):
        self.agent = agent
        self.model = model
        self.language = language
        self.prompts = prompts[language]

    def reformulate(self, query, nb_reformulation):
        prompt = self.prompts['query_reformulation']['QUERY_TEMPLATE'].format(query=query)
        system_prompt = self.prompts['query_reformulation']['SYSTEM_PROMPT'].format(language=self.language)
        prompts = []
        for i in range(nb_reformulation):
            prompts.append(prompt)
        queries = self.agent.multiple_predict(prompts=prompts, system_prompt=system_prompt, model=self.model)
        nb_input_tokens = queries['nb_input_tokens']
        nb_output_tokens = queries['nb_output_tokens']
        impacts = [0, 0, queries['impacts'][2]]
        energy = [0, 0, queries['energy'][2]]
        impacts[0] += queries['impacts'][0]
        impacts[1] += queries['impacts'][1]
        energy[0] += queries['energy'][0]
        energy[1] += queries['energy'][1]
        queries = queries['texts']
        queries.append(query)
        queries = remove_duplicates(queries)
        return (queries, nb_input_tokens, nb_output_tokens, impacts, energy)