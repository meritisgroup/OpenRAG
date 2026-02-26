from dataclasses import dataclass, field
from typing import Type, List

from methods.graph_rag.agent import GraphRagAgent
from methods.naive_rag.agent import NaiveRagAgent
from methods.agentic_rag.agent import AgenticRagAgent
from methods.merger_rag.agent import MergerRagAgent
from methods.query_based_rag.agent import QueryBasedRagAgent
from methods.self_rag.agent import SelfRagAgent
from methods.corrective_rag.agent import CragAgent
from methods.query_reformulation.agent import QueryReformulationRag
from methods.reranker_rag.agent import RerankerRag
from methods.semantic_chunking_rag.agent import SemanticChunkingRagAgent
from methods.contextual_retrieval_rag.agent import ContextualRetrievalRagAgent
from methods.advanced_rag.agent import AdvancedRag
from methods.agentic_rag_router.agent import AgenticRouterRAG
from methods.naive_chatbot.agent import NaiveChatbot


@dataclass
class RAGConfig:
    agent_class: Type
    requires_database: bool = True
    supported_in_custom: bool = True
    aliases: List[str] = field(default_factory=list)


RAG_REGISTRY: dict[str, RAGConfig] = {
    'naive': RAGConfig(
        agent_class=NaiveRagAgent,
        aliases=['naive_rag']
    ),
    'agentic': RAGConfig(
        agent_class=AgenticRagAgent,
        aliases=['agentic_rag']
    ),
    'agentic_router': RAGConfig(
        agent_class=AgenticRouterRAG,
        aliases=['agentic_rag_router']
    ),
    'merger': RAGConfig(
        agent_class=MergerRagAgent,
        aliases=['merger_rag']
    ),
    'naive_chatbot': RAGConfig(
        agent_class=NaiveChatbot,
        requires_database=False,
        supported_in_custom=False
    ),
    'reranker_rag': RAGConfig(
        agent_class=RerankerRag,
        aliases=['reranker']
    ),
    'advanced_rag': RAGConfig(
        agent_class=AdvancedRag,
        aliases=['advanced']
    ),
    'query_reformulation_rag': RAGConfig(
        agent_class=QueryReformulationRag,
        aliases=['query_reformulation']
    ),
    'graph': RAGConfig(
        agent_class=GraphRagAgent,
        aliases=['graph_rag']
    ),
    'query_based': RAGConfig(
        agent_class=QueryBasedRagAgent,
        aliases=['query_based_rag']
    ),
    'self': RAGConfig(
        agent_class=SelfRagAgent,
        aliases=['self_rag']
    ),
    'crag': RAGConfig(
        agent_class=CragAgent,
        aliases=['corrective_rag']
    ),
    'semantic_chunking': RAGConfig(
        agent_class=SemanticChunkingRagAgent,
        aliases=['semantic_chunking_rag']
    ),
    'contextual_retrieval': RAGConfig(
        agent_class=ContextualRetrievalRagAgent,
        aliases=['contextual_retrieval_rag']
    ),
}
