from utils.agent import Agent
from base_classes import Search
from database.database_class import DataBase
from methods.naive_rag.indexation import contexts_to_prompts
from database.rag_classes import Chunk_query

class QbSearch(Search):

    def __init__(self, agent: Agent, data_manager, nb_questions: int=10, language: str='EN') -> None:
        super().__init__(agent)
        self.data_manager = data_manager
        self.nb_questions = nb_questions
        self.language = language
        self.data_manager = data_manager

    def get_context(self, query: str) -> list[list[Chunk_query]]:
        if type(query) is str:
            query = [query]
        search_res = self.data_manager.k_search(queries=query, k=self.nb_questions, type_output=Chunk_query)
        return search_res