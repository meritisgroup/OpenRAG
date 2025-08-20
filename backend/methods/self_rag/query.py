from ...utils.agent import Agent
from ...base_classes import Search


class NaiveSearch(Search):
    def __init__(self, data_manager, nb_chunks: int = 10) -> None:
        """
        Args:
            vector_base (VectorBase): Vector base containing embeddings of chunks
            nb_chunks (int): Top-k chunks you want to add in context
        """
        super().__init__(Agent)
        self.data_manager = data_manager
        self.nb_chunks = nb_chunks

    def get_context(self, query: str) -> str:
        """
        Build the context using naive rag method.

        Args :
            query (str): user's request you want a context to answer
        """
        search_res = self.data_manager.k_search(
            queries=query,
            k=self.nb_chunks,
            output_fields=["text"],
        )
        docs_name = [res["doc_name"] for res in search_res[0]]
        chunks = [res["text"] for res in search_res[0]]

        return chunks, docs_name
