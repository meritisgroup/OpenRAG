import json
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from .abstract_class import AgentResponse, AgentType, QueryType, QueryPlan
from .config import AgenticRouterConfig
from methods.naive_rag.query import NaiveSearch
from methods.advanced_rag.reranker import Reranker
from utils.agent_functions import get_system_prompt
from .prompts import prompts
from .utils import ConfidenceParser
import numpy as np
import logging

# Configuration du logging
logger = logging.getLogger(__name__)

class BaseAgent(ABC):

    def __init__(self, config_server, agent, agent_type: AgentType, language: str='FR'):
        self.config_server = config_server
        self.language = language
        self.agent = agent
        self.agent_type = agent_type
        self.llm_model = self.config_server['model']
        self.nb_input_tokens = 0
        self.nb_output_tokens = 0
        self.prompts = prompts[self.language]
        self.system_prompt = get_system_prompt(self.config_server, self.prompts)

    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> AgentResponse:
        pass

class RouterAgent(BaseAgent):

    def __init__(self, agent, config_server, language: str='FR'):
        super().__init__(config_server, agent, AgentType.ROUTER, language)

    def process(self, input_data: str, **kwargs) -> AgentResponse:
        """
        Traite la requête et génère un plan d'exécution.
        """
        try:
            routing_prompt = self.prompts['routing']['QUERY_TEMPLATE'].format(input_data=input_data)
            system_prompt = self.prompts['routing']['SYSTEM_PROMPT']

            response = self.agent.predict_json(
                prompt=routing_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                model=self.llm_model,
                json_format=QueryPlan
            )

            return (
                AgentResponse(
                    content=response,
                    confidence=response.confidence_threshold,
                    reasoning=response.requires_reasoning,
                    metadata=response.model_dump()
                ),
                0,
                0
            )
        except Exception as e:
            logger.error(f"Erreur lors du routage: {str(e)}")
            # Plan par défaut robuste
            default_plan = QueryPlan(
                query_type='factual',
                sub_queries=[input_data],
                retrieval_strategy='dense',
                nb_chunks_per_query=AgenticRouterConfig.FALLBACK_NB_CHUNKS,
                requires_reasoning=False,
                requires_synthesis=False,
                confidence_threshold=AgenticRouterConfig.CONFIDENCE_THRESHOLD
            )
            return (
                AgentResponse(
                    content=default_plan,
                    confidence=0.5,
                    reasoning=f"Erreur d'analyse, utilisation du plan par défaut",
                    metadata={'error': str(e), 'fallback': True}
                ),
                0,
                0
            )

class RetrieverAgent(BaseAgent):

    def __init__(self, config_server, agent, data_manager, reranker: Optional[Reranker]=None, language: str='FR'):
        super().__init__(config_server, agent, AgentType.RETRIEVER, language)
        self.data_manager = data_manager
        self.reranker = reranker

    def process(self, input_data: Dict, **kwargs) -> AgentResponse:
        query = input_data['query']
        strategy = input_data.get('strategy', 'dense')
        nb_chunks = input_data.get('nb_chunks', 5)
        ns = NaiveSearch(data_manager=self.data_manager, nb_chunks=nb_chunks * 2)
        chunk_lists = ns.get_context(query=query)
        chunk_list = [chunk for sublist in chunk_lists for chunk in sublist]
        if self.reranker and len(chunk_list) > 0:
            (reranked_chunks, metadata, nb_tokens) = self.reranker.rerank(query=query, chunk_list=chunk_list, max_contexts=nb_chunks)
            self.nb_input_tokens += nb_tokens
            confidence = 0.85
        else:
            reranked_chunks = chunk_list[:nb_chunks]
            confidence = 0.7
        return AgentResponse(content=reranked_chunks, confidence=confidence, reasoning=f'Récupération de {len(reranked_chunks)} chunks avec stratégie {strategy}', metadata={'total_chunks': len(chunk_list), 'selected_chunks': len(reranked_chunks), 'strategy': strategy})

class ReasonerAgent(BaseAgent):

    def __init__(self, config_server, agent, language: str='FR'):
        super().__init__(config_server, agent, AgentType.REASONER, language=language)
        self.language = language

    def process(self, input_data: Dict, **kwargs) -> AgentResponse:
        """
        Traite une requête avec raisonnement sur les chunks fournis.
        """
        try:
            query = input_data['query']
            chunks = input_data['chunks']
            max_chunks = input_data.get('max_chunks', len(chunks))

            if not chunks:
                return (
                    AgentResponse(
                        content="Je n'ai pas trouvé d'informations pertinentes pour répondre à cette question.",
                        confidence=AgenticRouterConfig.DEFAULT_CONFIDENCE * 0.3,  # Plus faible si pas de chunks
                        reasoning='Aucun chunk disponible pour le raisonnement',
                        metadata={'nb_chunks_analyzed': 0, 'error': 'no_chunks'}
                    ),
                    0,
                    0
                )

            # Limiter le nombre de chunks analysés
            chunks = chunks[:max_chunks]

            context_text = '\n\n'.join([
                f'[Document {i + 1}]\n{chunk.text}'
                for (i, chunk) in enumerate(chunks)
            ])

            reasoning_prompt = self.prompts['reasonning']['QUERY_TEMPLATE'].format(
                query=query,
                context_text=context_text
            )
            system_prompt = self.prompts['reasonning']['SYSTEM_PROMPT']

            response = self.agent.predict(
                prompt=reasoning_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                model=self.llm_model
            )

            input_tokens = response.get('nb_input_tokens', 0)
            output_tokens = response.get('nb_output_tokens', 0)
            text = response['texts']

            # Utiliser le parser de confiance robuste
            confidence = ConfidenceParser.parse_confidence(text, default=0.6)

            return (
                AgentResponse(
                    content=text,
                    confidence=confidence,
                    reasoning='Raisonnement Chain-of-Thought appliqué',
                    metadata={
                        'nb_chunks_analyzed': len(chunks),
                        'max_chunks': max_chunks,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens
                    }
                ),
                input_tokens,
                output_tokens
            )

        except Exception as e:
            logger.error(f"Erreur lors du raisonnement: {str(e)}")
            return (
                AgentResponse(
                    content=f"Erreur lors du traitement de la question: {str(e)}",
                    confidence=0.1,
                    reasoning='Erreur de traitement',
                    metadata={'error': str(e), 'nb_chunks_analyzed': len(input_data.get('chunks', []))}
                ),
                0,
                0
            )

class SynthesizerAgent(BaseAgent):

    def __init__(self, config_server, agent, language: str='FR'):
        super().__init__(config_server, agent, AgentType.SYNTHESIZER, language=language)
        self.language = language

    def process(self, input_data: Dict, **kwargs) -> AgentResponse:
        """
        Synthétise les réponses partielles en une réponse cohérente.
        """
        try:
            query = input_data['query']
            sub_responses = input_data['sub_responses']

            synthesis_prompt = self.prompts['synthesis']['QUERY_TEMPLATE'].format(
                query=query,
                answers=self._format_sub_responses(sub_responses)
            )
            system_prompt = self.prompts['synthesis']['SYSTEM_PROMPT']

            response = self.agent.predict(
                prompt=synthesis_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                model=self.llm_model
            )

            input_tokens = response.get('nb_input_tokens', 0)
            output_tokens = response.get('nb_output_tokens', 0)

            # Calculer la confiance moyenne avec validation
            confidences = [
                AgenticRouterConfig.validate_confidence(r.get('confidence', AgenticRouterConfig.DEFAULT_CONFIDENCE))
                for r in sub_responses
            ]
            avg_confidence = AgenticRouterConfig.validate_confidence(
                np.mean(confidences) if confidences else AgenticRouterConfig.DEFAULT_CONFIDENCE
            )

            return (
                AgentResponse(
                    content=response['texts'],
                    confidence=avg_confidence,
                    reasoning='Synthèse de multiples sources',
                    metadata={
                        'nb_sources': len(sub_responses),
                        'confidences': confidences,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens
                    }
                ),
                input_tokens,
                output_tokens
            )

        except Exception as e:
            logger.error(f"Erreur lors de la synthèse: {str(e)}")
            return (
                AgentResponse(
                    content="Erreur lors de la synthèse des réponses.",
                    confidence=AgenticRouterConfig.DEFAULT_CONFIDENCE * 0.3,
                    reasoning='Erreur de synthèse',
                    metadata={'error': str(e), 'nb_sources': len(input_data.get('sub_responses', []))}
                ),
                0,
                0
            )

    def _format_sub_responses(self, sub_responses: List[Dict]) -> str:
        formatted = []
        for (i, resp) in enumerate(sub_responses, 1):
            formatted.append(f"\n--- Réponse {i} (confiance: {resp.get('confidence', 'N/A')}) ---\n{resp.get('content', 'N/A')}\n")
        return '\n'.join(formatted)

class EvaluatorAgent(BaseAgent):

    def __init__(self, config_server, agent, language: str='FR'):
        super().__init__(config_server, agent, AgentType.EVALUATOR, language=language)
        self.language = language

    def process(self, input_data: Dict, **kwargs) -> AgentResponse:
        """
        Évalue la qualité de la réponse générée.
        """
        try:
            query = input_data['query']
            answer = input_data['answer']
            chunks = input_data.get('chunks', [])

            eval_prompt = self.prompts['evaluate']['QUERY_TEMPLATE'].format(
                query=query,
                answer=answer
            )
            system_prompt = self.prompts['evaluate']['SYSTEM_PROMPT']

            response = self.agent.predict(
                prompt=eval_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                model=self.llm_model
            )

            input_tokens = response.get('nb_input_tokens', 0)
            output_tokens = response.get('nb_output_tokens', 0)

            # Utiliser le parser JSON robuste
            eval_data = ConfidenceParser.safe_parse_json(response['texts'])

            if not eval_data:
                # Fallback si le parsing échoue
                eval_data = {
                    'score_global': AgenticRouterConfig.DEFAULT_CONFIDENCE,
                    'needs_improvement': False,
                    'reasoning': 'Évaluation par défaut (parsing échoué)',
                    'suggestions': ''
                }

            return (
                AgentResponse(
                    content=eval_data,
                    confidence=eval_data.get('score_global', 0.5),
                    reasoning=eval_data.get('reasoning', ''),
                    metadata={
                        'needs_improvement': eval_data.get('needs_improvement', False),
                        'suggestions': eval_data.get('suggestions', ''),
                        'parsing_success': bool(eval_data.get('score_global'))
                    }
                ),
                input_tokens,
                output_tokens
            )

        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation: {str(e)}")
            return (
                AgentResponse(
                    content={
                        'score_global': AgenticRouterConfig.DEFAULT_CONFIDENCE,
                        'needs_improvement': False,
                        'error': str(e)
                    },
                    confidence=AgenticRouterConfig.DEFAULT_CONFIDENCE,
                    reasoning='Évaluation par défaut (erreur)',
                    metadata={'error': str(e), 'parsing_success': False}
                ),
                0,
                0
            )