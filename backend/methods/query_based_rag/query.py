from ...utils.agent import Agent
from ...base_classes import Search
from ...database.database_class import DataBase
from ..naive_rag.indexation import contexts_to_prompts


class QbSearch(Search):
    def __init__(
        self,
        agent: Agent,
        data_manager,
        nb_questions: int = 10,
        language: str = "EN",
    ) -> None:
        
        super().__init__(agent)
        self.data_manager = data_manager
        self.nb_questions = nb_questions
        self.language = language
        self.data_manager = data_manager


    def get_context(self, query: str) -> str:
        search_res = self.data_manager.k_search(
            queries=query,
            k=self.nb_questions,
            output_fields=["text", "doc_name", "chunk_text"],
        )
        chunks_id = sorted(
            [(res["doc_name"]) for res in search_res[0]],
            key=lambda x: x[0],
        )
        chunks = [res["text"] for res in search_res[0]]
        docs_name = [res["doc_name"] for res in search_res[0]]
        context = ""
        context, docs_name = contexts_to_prompts(contexts=chunks,
                                                 docs_name=docs_name)

        return context, docs_name
