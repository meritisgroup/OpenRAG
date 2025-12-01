from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pydantic import BaseModel

class QueryType(Enum):
    """Types de requêtes possibles"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    CREATIVE = "creative"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"


class AgentType(Enum):
    """Types d'agents spécialisés"""
    ROUTER = "router"
    RETRIEVER = "retriever"
    REASONER = "reasoner"
    SYNTHESIZER = "synthesizer"
    EVALUATOR = "evaluator"
    PLANNER = "planner"


class QueryPlan(BaseModel):
    """Plan d'exécution pour une requête"""
    query_type: str
    sub_queries: List[str]
    retrieval_strategy: str
    nb_chunks_per_query: int
    requires_reasoning: bool
    requires_synthesis: bool
    confidence_threshold: float

@dataclass
class AgentResponse:
    """Réponse d'un agent"""
    content: Any
    confidence: float
    reasoning: str
    metadata: Dict


