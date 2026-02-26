from abc import ABC, abstractmethod
from utils.agent import Agent
from typing import List, Dict, Any, Optional
import concurrent.futures
from utils.threading_utils import get_executor_threads

class Indexation(ABC):

    @abstractmethod
    def run_pipeline(self, data_path, storage_path, *kwargs) -> None:
        pass

class RagAgent(ABC):
    config_server: Dict[str, Any]
    nb_input_tokens: int
    nb_output_tokens: int

    @abstractmethod
    def indexation_phase(self, reset_index: bool=False, reset_preprocess: bool=False, overlap: bool=True, **kwargs) -> None:
        pass

    @abstractmethod
    def generate_answer(self, query: str, nb_chunks: int=5, options_generation: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        pass

    def generate_answers(self, queries: List[str], nb_chunks: int=5, options_generation: Optional[Dict[str, Any]]=None, max_workers: Optional[int]=None) -> List[Dict[str, Any]]:
        if max_workers is None:
            max_workers = self.config_server.get('max_workers', 10)
        if max_workers <= get_executor_threads():
            max_workers = 1
        answers = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {executor.submit(self.generate_answer, query=query, nb_chunks=nb_chunks, options_generation=options_generation): query for query in queries}
            for future in concurrent.futures.as_completed(future_to_query):
                try:
                    result = future.result()
                    answers.append(result)
                    if result:
                        self.nb_input_tokens += result.get('nb_input_tokens', 0)
                        self.nb_output_tokens += result.get('nb_output_tokens', 0)
                except Exception as exc:
                    query_that_failed = future_to_query[future]
        query_map = {query: i for (i, query) in enumerate(queries)}
        answers.sort(key=lambda x: query_map.get(x.get('original_query', ''), 0))
        return answers

class Search(ABC):

    def __init__(self, agent: Agent):
        self.agent = agent
        pass

    @abstractmethod
    def get_context(self, query: str) -> str:
        pass