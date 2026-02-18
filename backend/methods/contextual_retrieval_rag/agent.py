from ...base_classes import RagAgent
from ...utils.agent import get_Agent
from ...utils.factory_vectorbase import get_vectorbase
from ...utils.agent_functions import get_system_prompt
from .prompts import prompts
from ..advanced_rag.agent import AdvancedRag
from .indexation import ContextualRetrievalIndexation
from ..naive_rag.query import NaiveSearch
import numpy as np
from sqlalchemy import func
from backend.database.rag_classes import Document, Tokens
from ..query_reformulation.query_reformulation import query_reformulation
from ...database.database_class import get_management_data


class ContextualRetrievalRagAgent(AdvancedRag):
    """
    For each chunk an LLM is asked to read the whole document and to generate in a short paragraph / sentence explaining the chunk given the context.
    """

    def __init__(
        self,
        config_server: dict,
        models_infos: dict,
        dbs_name: list[str],
        data_folders_name: list[str]
    ) -> None:
        """
        Args:
            model (str): model used to generate context, to be set in backend/methods/contextual_retrieval_rag/config.json file
            storage_path: folder in which database will be stored
            params_host_llm(dict): parameters for Ollama or VLLM, to be set in backend/config_server.json file
            params_vectorbase(dict): vectorbase connection parameters, to be set in backend/config_server.json file
            embedding_model (str): Model used to embed documents and queries, to be set in backend/methods/contextual_retrieval_rag/config.json file
            language (str) : Sets the language of the prompts (available "EN", "FR"), to be set in in backend/methods/contextual_retrieval_rag/config.json file
            api_key (str) : API key to be used if needed, to be set in backend/config_server.json file (not mandatory if using Ollama or VLLM)
            db_name (str) : Name given to the database that keeps track of already processed docs, if it already exists adds new documents to the existing database (stored in storage/ folder)
            vb_name (str) : Name given to the vectorbase, if it already exists adds new documents to the existing vectorbase (stored in milvus/elasticsearch docker)
        """

        config_server["ProcessorChunks"] = ["Contextual"]
        config_server["reranker_model"] = None
        config_server["reformulate_query"] = False

        super().__init__(
            config_server=config_server,
            models_infos=models_infos,
            dbs_name=dbs_name,
            data_folders_name=data_folders_name,
        )

        self.system_prompt = get_system_prompt(self.config_server, self.prompts)    

