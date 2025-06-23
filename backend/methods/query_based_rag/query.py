from ...utils.agent import Agent
from ...base_classes import Search
from ...database.database_class import DataBase


class QbSearch(Search):
    def __init__(
        self,
        agent: Agent,
        data_base: DataBase,
        vector_base,
        nb_questions: int = 10,
        language: str = "EN",
    ) -> None:
        super().__init__(agent)
        self.db = data_base
        self.vb = vector_base
        self.nb_questions = nb_questions
        self.language = language

    def get_context(self, query: str) -> str:
        search_res = self.vb.k_search(
            queries=query,
            collection_name=self.db.name[:-3],
            k=self.nb_questions,
            output_fields=["text", "doc_name", "chunk_text"],
        )
        chunks_ids = sorted(
            [(res["doc_name"]) for res in search_res[0]],
            key=lambda x: x[0],
        )

        context = ""
        for i in range(len(search_res[0])):
            if "chunk_text" in search_res[0][i].keys():
                if search_res[0][i]["chunk_text"] not in context:
                    context += search_res[0][i]["chunk_text"] + "\n[...]\n"

        context = context[:-7]

        return context
