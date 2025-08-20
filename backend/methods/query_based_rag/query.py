from ...utils.agent import Agent
from ...base_classes import Search
from ...database.database_class import DataBase


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

        context = ""
        docs_name = []
        for i in range(len(search_res[0])):
            if "chunk_text" in search_res[0][i].keys():
                if search_res[0][i]["chunk_text"] not in context:
                    context += search_res[0][i]["chunk_text"] + "\n[...]\n"
                    docs_name.append(search_res[0]["docs_name"])

        context = context[:-7]

        return context, docs_name
