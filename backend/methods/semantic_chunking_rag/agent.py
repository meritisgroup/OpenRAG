from ...base_classes import RagAgent
from ...utils.factory_vectorbase import get_vectorbase
from ...utils.agent_functions import get_system_prompt
from ...database.database_class import get_management_data
from ...utils.agent import get_Agent
from ..advanced_rag.agent import AdvancedRag
from .prompts import prompts
import numpy as np
from ..query_reformulation.query_reformulation import query_reformulation


class SemanticChunkingRagAgent(AdvancedRag):
    """
    This RAG methods uses an adaptative size of chunk in order to group sentences by similarity
    """

    def __init__(
        self,
        config_server: dict,
        dbs_name: list[str],
        data_folders_name: list[str]
    ) -> None:
        """
        Args:
            model (str): model used to generate context, to be set in backend/methods/semantic_chunking_rag/config.json file
            storage_path: folder in which database will be stored
            params_vectorbase(dict): vectorbase connection parameters, to be set in backend/config_server.json file
            params_host_llm(dict): parameters for Ollama or VLLM, to be set in backend/config_server.json file
            language (str) : Sets the language of the prompts (available "EN", "FR"), to be set in in backend/methods/semantic_chunking_rag/config.json file
            api_key (str) : API key to be used if needed, to be set in backend/config_server.json file (not mandatory if using Ollama or VLLM)
            db_name (str) : Name given to the database that keeps track of already processed docs, if it already exists adds new documents to the existing database (stored in storage/ folder)
            vb_name (str) : Name given to the vectorbase, if it already exists adds new documents to the existing vectorbase (stored in milvus/elasticsearch docker)
            embedding_model (str) : Model used to embed documents and queries, to be set in backend/methods/semantic_chunking_rag/config.json file
            type_retrieval (str) : How documents will be retrieved (embeddings, BM25, vlm_embeddings are available plus hybrid if using elasticsearch)

        Returns:
            None
        """
        config_server["TextSplitter"] = "Semantic_TextSplitter"
        config_server["ProcessorChunks"] = []
        config_server["reranker_model"] = None
        config_server["reformulate_query"] = False
        
        super().__init__(
            config_server=config_server,
            dbs_name=dbs_name,
            data_folders_name=data_folders_name,
        )
        self.system_prompt = get_system_prompt(self.config_server, self.prompts)  