from ..naive_rag.query import NaiveSearch
from ..advanced_rag.agent import AdvancedRag
from itertools import chain
from .prompts import prompts
from .query_reformulation import query_reformulation
from ..naive_rag.indexation import contexts_to_prompts
import numpy as np


class QueryReformulationRag(AdvancedRag):
    def __init__(
        self,
        config_server: dict,
        dbs_name: list[str],
        data_folders_name: list[str]
    ) -> None:

        config_server["ProcessorChunks"] = []
        config_server["reranker_model"] = None
        config_server["reformulate_query"] = True

        super().__init__(
            config_server=config_server,
            dbs_name=dbs_name,
            data_folders_name=data_folders_name
        )
        self.nb_chunks = config_server["nb_chunks"]
        self.prompts = prompts[self.language]