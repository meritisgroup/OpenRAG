from utils.agent import Agent
from base_classes import Search
from methods.naive_rag.indexation import contexts_to_prompts

class NaiveSearch(Search):

    def __init__(self, vector_base, nb_chunks: int=10) -> None:
        super().__init__(Agent)
        self.vb = vector_base
        self.nb_chunks = nb_chunks

    def get_context(self, query: str) -> str:
        if type(query) is str:
            query = [query]
        search_res = self.vb.k_search(queries=query, k=self.nb_chunks, output_fields=['text', 'doc_name'])
        chunks = [res['text'] for res in search_res[0]]
        docs_name = [res['doc_name'] for res in search_res[0]]
        context = contexts_to_prompts(context=chunks, docs_name=docs_name)
        return (context, docs_name)