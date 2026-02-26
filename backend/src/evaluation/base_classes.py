from abc import ABC, abstractmethod
from utils.agent import Agent
from pydantic import BaseModel

class Evaluator(ABC):

    def __init__(self, agent: Agent, model: str, max_attempts=5, batch_size: int=10) -> None:
        super().__init__()
        self.agent = agent
        self.model = model
        self.language = agent.language
        self.max_attempts = max_attempts

    @abstractmethod
    def run_evaluation_pipeline(self):
        pass

class MetricAnswer(BaseModel):
    winner: str

class GroundTruthAnswer(BaseModel):
    score: int

class ChunkRelevanceAnswer(BaseModel):
    score: int

class ContextFaithfulnessAnswer(BaseModel):
    statements: list[str]

class StatementSupported(BaseModel):
    supported: bool