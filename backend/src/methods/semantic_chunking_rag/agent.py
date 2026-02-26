from methods.advanced_rag.agent import AdvancedRag
from .prompts import prompts
import numpy as np

class SemanticChunkingRagAgent(AdvancedRag):

    def __init__(self, config_server: dict, models_infos: dict, dbs_name: list[str], data_folders_name: list[str]) -> None:
        config_server['TextSplitter'] = 'Semantic_TextSplitter'
        config_server['ProcessorChunks'] = []
        config_server['reranker_model'] = None
        config_server['reformulate_query'] = False
        super().__init__(config_server=config_server, models_infos=models_infos, dbs_name=dbs_name, data_folders_name=data_folders_name)
        self.system_prompt = self._get_system_prompt(self.prompts)