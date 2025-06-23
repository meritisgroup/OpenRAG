from abc import ABC, abstractmethod
from .utils.agent import Agent
from typing import List


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
