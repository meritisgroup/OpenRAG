from abc import ABC, abstractmethod
from .utils.agent import Agent
from typing import List
import concurrent.futures
from .utils.threading_utils import get_executor_threads

class Indexation(ABC):

    @abstractmethod
    def run_pipeline(self, data_path, storage_path, *kwargs) -> None:
        """
        Run all the part of indexation to build either a database and/or fill the Vectorbase which will be used for the generation part
        """
        pass


class RagAgent(ABC):

    @abstractmethod
    def indexation_phase(self, data_pth, **kwargs) -> None:
        """Handles the indexation phase with flexible parameters.

        Args:
            path_input (str): Path to the input data.
            **kwargs: Additional parameters for specific implementations.
        """
        pass

    @abstractmethod
    def generate_answer(
        self, query: str, model: str, method_parameter: str = 2, **kwargs
    ) -> str:
        """Generates RAG context.
        Args:
            query (str): The query string.
            method_parameter (int): Parameter for the method. Default is 2.
            model (str, optional): Model to use.
            **kwargs: Additional configuration.
        """
        pass

    def generate_answers(self,
                         queries: list[str], 
                         nb_chunks: int = 2, 
                         options_generation=None,
                         max_workers = None):
        
        if max_workers is None:
            max_workers = self.config_server["max_workers"]
        
        if max_workers<=get_executor_threads():
            max_workers = 1

        answers = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {
                executor.submit(
                    self.generate_answer, 
                    query=query, 
                    nb_chunks=nb_chunks, 
                    options_generation=options_generation
                ): query for query in queries
            }
            for future in concurrent.futures.as_completed(future_to_query):
                try:
                    result = future.result()
                    answers.append(result)
                    
                    self.nb_input_tokens += result["nb_input_tokens"]
                    self.nb_output_tokens += result["nb_output_tokens"]

                except Exception as exc:
                    query_that_failed = future_to_query[future]
                   
        query_map = {query: i for i, query in enumerate(queries)}
        answers.sort(key=lambda x: query_map[x['original_query']])
        return answers


class Search(ABC):

    def __init__(self, agent: Agent):
        """Initialize the Search class with an Agent

        Args:
            agent (Agent): The agent responsible for predictions and task handling."""
        self.agent = agent
        pass

    @abstractmethod
    def get_context(self, query: str) -> str:
        pass
