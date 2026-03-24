from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from methods.advanced_rag.indexation import AdvancedIndexation
from methods.naive_rag.indexation import contexts_to_prompts
from methods.advanced_rag.agent import AdvancedRag
from methods.naive_rag.query import NaiveSearch
from methods.advanced_rag.reranker import Reranker
from methods.query_reformulation.query_reformulation import query_reformulation
from .agents_router import RouterAgent, RetrieverAgent, ReasonerAgent, SynthesizerAgent, EvaluatorAgent
from .abstract_class import QueryPlan
from .config import AgenticRouterConfig
from database.rag_classes import Chunk
from core.error_handler import LLMError, RetrievalError

class AgenticRouterRAG(AdvancedRag):

    def __init__(self, config_server: dict, models_infos: dict, dbs_name: list[str], data_folders_name: list[str]) -> None:
        super().__init__(config_server=config_server, models_infos=models_infos, dbs_name=dbs_name, data_folders_name=data_folders_name)

        # Configuration centralisée
        self.max_iterations = AgenticRouterConfig.MAX_ITERATIONS
        self.enable_self_correction = True
        self.confidence_threshold = AgenticRouterConfig.CONFIDENCE_THRESHOLD

        if self.reranker_model:
            self.reranker = Reranker(agent=self.agent, reranking_model=self.reranker_model)
        else:
            self.reranker = None
        self._initialize_agents()
        self.execution_history = []

    def _initialize_agents(self):
        self.router_agent = RouterAgent(config_server=self.config_server, agent=self.agent, language=self.language)
        self.retriever_agent = RetrieverAgent(config_server=self.config_server, agent=self.agent, data_manager=self.data_manager, reranker=self.reranker, language=self.language)
        self.reasoner_agent = ReasonerAgent(config_server=self.config_server, agent=self.agent, language=self.language)
        self.synthesizer_agent = SynthesizerAgent(config_server=self.config_server, agent=self.agent, language=self.language)
        self.evaluator_agent = EvaluatorAgent(config_server=self.config_server, agent=self.agent, language=self.language)

    def indexation_phase(self, reset_index: bool=False, reset_preprocess: bool=False, overlap: bool=True) -> None:
        if reset_preprocess:
            reset_index = True
        if reset_index:
            self.data_manager.delete_collection()
            self.data_manager.clean_database()
        index = AdvancedIndexation(data_manager=self.data_manager, type_text_splitter=self.type_text_splitter, data_preprocessing=self.config_server['data_preprocessing'], agent=self.agent, embedding_model=self.embedding_model, llm_model=self.llm_model, type_processor_chunks=self.type_processor_chunks, language=self.language)
        index.run_pipeline(chunk_size=self.config_server.get('chunk_size', 512), chunk_overlap=overlap, batch=self.config_server.get('batch', 32), config_server=self.config_server, reset_preprocess=reset_preprocess)

    def generate_answer(self, query: str, nb_chunks: int=7, options_generation: Optional[Dict]=None, return_execution_trace: bool=False) -> Dict:
        execution_trace = []
        iteration = 0
        total_input_tokens = 0
        total_output_tokens = 0
        (impacts, energies) = ([0, 0, ''], [0, 0, ''])

        # Étape 1 : Routage et décomposition de la requête
        (router_response, input_tokens, output_tokens) = self.router_agent.process(query)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        query_plan: QueryPlan = router_response.content

        # Chaque sous-requête récupère le même nombre de chunks que le total demandé
        # Ensuite on déduplique et on filtre pour avoir les meilleurs chunks au final
        query_plan.nb_chunks_per_query = nb_chunks

        execution_trace.append({
            'step': 'routing',
            'response': router_response,
            'plan': query_plan,
            'nb_chunks_requested': nb_chunks,
            'nb_sub_queries': len(query_plan.sub_queries),
            'chunks_per_sub_query': nb_chunks,
            'strategy': 'over_retrieve_then_filter'
        })

        # Ajouter la requête originale si elle n'est pas déjà présente
        if query not in query_plan.sub_queries:
            query_plan.sub_queries.append(query)

        # Étape 2 : Traitement parallèle des sous-requêtes
        all_chunks, sub_responses, parallel_tokens = self._process_sub_queries_parallel(query_plan, query)
        total_input_tokens += parallel_tokens['input']
        total_output_tokens += parallel_tokens['output']

        # Déduplication des chunks (basé sur le contenu)
        nb_chunks_before_dedup = len(all_chunks)
        all_chunks = self._deduplicate_chunks(all_chunks)

        # Garder les nb_chunks meilleurs chunks (par ordre de récupération)
        if len(all_chunks) > nb_chunks:
            all_chunks = all_chunks[:nb_chunks]

        execution_trace.append({
            'step': 'retrieval_parallel',
            'nb_chunks_requested': nb_chunks,
            'nb_chunks_per_sub_query': query_plan.nb_chunks_per_query,
            'nb_chunks_before_dedup': nb_chunks_before_dedup,
            'nb_chunks_after_dedup': len(all_chunks),
            'nb_sub_queries': len(query_plan.sub_queries),
            'sub_responses': sub_responses,
            'parallel_processing': True,
            'deduplication_applied': True,
            'chunks_removed': nb_chunks_before_dedup - len(all_chunks)
        })

        # Étape 3 : Synthèse ou génération de réponse
        answer, confidence, synthesis_tokens = self._generate_final_answer(
            query, query_plan, all_chunks, sub_responses, options_generation, nb_chunks
        )
        total_input_tokens += synthesis_tokens['input']
        total_output_tokens += synthesis_tokens['output']

        execution_trace.append({'step': 'generation', 'answer': answer, 'confidence': confidence})

        # Étape 4 : Auto-correction si activée
        if self.enable_self_correction:
            answer, correction_tokens = self._self_correct_if_needed(query, answer, all_chunks, query_plan, iteration)
            total_input_tokens += correction_tokens['input']
            total_output_tokens += correction_tokens['output']
            execution_trace.append({'step': 'self_correction', 'improved': True})

        # Agréger les impacts et énergie
        self._aggregate_environmental_impacts(impacts, energies)

        result = {
            'answer': answer,
            'confidence': confidence,
            'nb_input_tokens': total_input_tokens,
            'nb_output_tokens': total_output_tokens,
            'context': all_chunks,
            'impacts': impacts,
            'energy': energies,
            'query_plan': {
                'type': query_plan.query_type,
                'sub_queries': query_plan.sub_queries,
                'strategy': query_plan.retrieval_strategy
            },
            'metadata': {
                'nb_chunks_requested': nb_chunks,
                'nb_chunks_retrieved': len(all_chunks),
                'nb_chunks_per_sub_query': query_plan.nb_chunks_per_query,
                'nb_sub_queries': len(query_plan.sub_queries),
                'total_chunks_retrieved_before_dedup': nb_chunks * len(query_plan.sub_queries),  # Approximation
                'requires_reasoning': query_plan.requires_reasoning,
                'requires_synthesis': query_plan.requires_synthesis,
                'parallel_processing': True,
                'deduplication_enabled': True
            },
            'execution_trace': execution_trace if return_execution_trace else None
        }
        return result

    def _process_sub_queries_parallel(self, query_plan: QueryPlan, original_query: str) -> tuple:
        """
        Traite les sous-requêtes en parallèle avec ThreadPoolExecutor.
        """
        all_chunks = []
        sub_responses = []
        total_input_tokens = 0
        total_output_tokens = 0

        with ThreadPoolExecutor(max_workers=AgenticRouterConfig.get_max_workers(len(query_plan.sub_queries))) as executor:
            # Soumettre toutes les tâches en parallèle
            future_to_query = {
                executor.submit(self._process_single_sub_query, sub_query, query_plan): sub_query
                for sub_query in query_plan.sub_queries
            }

            # Récupérer les résultats au fur et à mesure qu'ils terminent
            for future in future_to_query:
                try:
                    result = future.result(timeout=AgenticRouterConfig.TIMEOUT_PER_QUERY)
                    all_chunks.extend(result['chunks'])
                    sub_responses.append(result['response'])
                    total_input_tokens += result['input_tokens']
                    total_output_tokens += result['output_tokens']
                except Exception as e:
                    # En cas d'erreur, on utilise la requête originale
                    sub_query = future_to_query[future]
                    fallback_result = self._fallback_retrieval(sub_query, query_plan)
                    all_chunks.extend(fallback_result['chunks'])
                    sub_responses.append(fallback_result['response'])
                    sub_responses.append(fallback_result['response'])

        return (all_chunks, sub_responses, {'input': total_input_tokens, 'output': total_output_tokens})

    def _process_single_sub_query(self, sub_query: str, query_plan: QueryPlan) -> Dict:
        """
        Traite une seule sous-requête.

        Stratégie "over-retrieve then filter" :
        - Chaque sous-requête récupère nb_chunks_per_query chunks (ex: 7)
        - Ensuite on déduplique tous les chunks récupérés
        - Et on garde les meilleurs nb_chunks chunks au final

        Exemple : 7 chunks demandés avec 3 sous-requêtes
        - Sous-requête 1 : 7 chunks
        - Sous-requête 2 : 7 chunks
        - Sous-requête 3 : 7 chunks
        - Total : 21 chunks → déduplication → 7 chunks finaux
        """
        result = {
            'chunks': [],
            'response': {},
            'input_tokens': 0,
            'output_tokens': 0
        }

        # Récupération des chunks (même nombre que le total demandé)
        retrieval_response = self.retriever_agent.process({
            'query': sub_query,
            'strategy': query_plan.retrieval_strategy,
            'nb_chunks': query_plan.nb_chunks_per_query
        })
        chunks = retrieval_response.content
        result['chunks'] = chunks

        # Raisonnement si nécessaire
        if query_plan.requires_reasoning and len(chunks) > 0:
            (reasoning_response, input_tokens, output_tokens) = self.reasoner_agent.process({
                'query': sub_query,
                'chunks': chunks
            })
            result['response'] = {
                'query': sub_query,
                'content': reasoning_response.content,
                'confidence': reasoning_response.confidence,
                'chunks': chunks
            }
            result['input_tokens'] = input_tokens
            result['output_tokens'] = output_tokens
        else:
            result['response'] = {
                'query': sub_query,
                'chunks': chunks,
                'confidence': retrieval_response.confidence
            }

        return result

    def _fallback_retrieval(self, sub_query: str, query_plan: QueryPlan) -> Dict:
        """
        Fallback en cas d'erreur lors du traitement parallèle.
        """
        try:
            nb_chunks = AgenticRouterConfig.get_nb_chunks(query_plan.nb_chunks_per_query)
            ns = NaiveSearch(data_manager=self.data_manager, nb_chunks=nb_chunks)
            chunk_lists = ns.get_context(query=sub_query)
            chunks = [chunk for chunk_list in chunk_lists for chunk in chunk_list]

            return {
                'chunks': chunks[:nb_chunks],
                'response': {
                    'query': sub_query,
                    'chunks': chunks[:nb_chunks],
                    'confidence': AgenticRouterConfig.DEFAULT_CONFIDENCE,
                    'error': True
                }
            }
        except Exception as e:
            return {
                'chunks': [],
                'response': {
                    'query': sub_query,
                    'chunks': [],
                    'confidence': 0.0,
                    'error': True,
                    'error_message': str(e)
                }
            }

    def _generate_final_answer(self, query: str, query_plan: QueryPlan, all_chunks: List[Chunk], sub_responses: List[Dict], options_generation: Optional[Dict], nb_chunks: int) -> tuple:
        """
        Génère la réponse finale avec synthèse si nécessaire.

        Args:
            query: Question originale
            query_plan: Plan d'exécution
            all_chunks: Tous les chunks récupérés
            sub_responses: Réponses partielles
            options_generation: Options de génération
            nb_chunks: Nombre de chunks à utiliser

        Returns:
            Tuple (answer, confidence, token_counts)
        """
        total_input_tokens = 0
        total_output_tokens = 0

        if query_plan.requires_synthesis and len(sub_responses) > 1:
            # Synthèse des réponses partielles
            (synthesis_response, input_tokens, output_tokens) = self.synthesizer_agent.process({
                'query': query,
                'sub_responses': sub_responses
            })
            answer = synthesis_response.content
            confidence = synthesis_response.confidence
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        elif query_plan.requires_reasoning and len(sub_responses) > 0:
            # Utiliser la première réponse raisonnée
            if 'content' in sub_responses[0]:
                answer = sub_responses[0]['content']
                confidence = sub_responses[0]['confidence']
            else:
                # Fallback : raisonnement sur tous les chunks
                (reasoning_response, input_tokens, output_tokens) = self.reasoner_agent.process({
                    'query': query,
                    'chunks': all_chunks[:nb_chunks]  # Limiter au nombre de chunks demandé
                })
                answer = reasoning_response.content
                confidence = reasoning_response.confidence
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

        else:
            # Génération standard
            # Limiter les chunks au nombre demandé
            limited_chunks = all_chunks[:nb_chunks]
            prompt = self.build_final_prompt(limited_chunks, query)
            if options_generation is None:
                options_generation = self.config_server.get('options_generation', {})

            response = self.agent.predict(
                prompt=prompt,
                system_prompt=self.system_prompt,
                options_generation=options_generation,
                model=self.llm_model
            )
            answer = response['texts']
            # Calculer la confiance basée sur les réponses partielles
            confidence = self._calculate_aggregated_confidence(sub_responses)
            total_input_tokens += response.get('nb_input_tokens', 0)
            total_output_tokens += response.get('nb_output_tokens', 0)

        return (answer, confidence, {'input': total_input_tokens, 'output': total_output_tokens})

    def _calculate_aggregated_confidence(self, sub_responses: List[Dict]) -> float:
        """
        Calcule la confiance agrégée à partir des réponses partielles.
        """
        if not sub_responses:
            return AgenticRouterConfig.DEFAULT_CONFIDENCE

        confidences = [
            AgenticRouterConfig.validate_confidence(r.get('confidence', AgenticRouterConfig.DEFAULT_CONFIDENCE))
            for r in sub_responses if r.get('confidence', 0) > 0
        ]

        if not confidences:
            return AgenticRouterConfig.DEFAULT_CONFIDENCE

        # Moyenne pondérée par le nombre de chunks
        weighted_confidences = []
        for response in sub_responses:
            conf = AgenticRouterConfig.validate_confidence(response.get('confidence', AgenticRouterConfig.DEFAULT_CONFIDENCE))
            chunks = response.get('chunks', [])
            weight = len(chunks) if chunks else 1
            weighted_confidences.append(conf * weight)

        total_weight = np.sum([len(r.get('chunks', [1])) for r in sub_responses])
        if total_weight == 0:
            return AgenticRouterConfig.DEFAULT_CONFIDENCE

        return AgenticRouterConfig.validate_confidence(np.sum(weighted_confidences) / total_weight)

    def _self_correct_if_needed(self, query: str, answer: str, chunks: List[Chunk], query_plan: QueryPlan, iteration: int) -> tuple:
        """
        Auto-correction de la réponse si nécessaire.
        """
        try:
            (eval_response, input_tokens, output_tokens) = self.evaluator_agent.process({
                'query': query,
                'answer': answer,
                'chunks': chunks
            })

            evaluation = eval_response.content

            if (evaluation.get('needs_improvement', False) and
                evaluation.get('score_global', 1.0) < query_plan.confidence_threshold and
                iteration < self.max_iterations):

                correction_prompt = self.prompts['correction']['QUERY_TEMPLATE'].format(
                    query=query,
                    answer=answer,
                    evaluation=evaluation.get('suggestions', 'Qualité insuffisante'),
                    context=self._format_chunks(chunks)
                )
                system_prompt = self.prompts['correction']['SYSTEM_PROMPT']

                corrected_response = self.agent.predict(
                    prompt=correction_prompt,
                    system_prompt=system_prompt,
                    model=self.llm_model
                )

                return (corrected_response['texts'], {
                    'input': input_tokens + corrected_response.get('nb_input_tokens', 0),
                    'output': output_tokens + corrected_response.get('nb_output_tokens', 0)
                })
        except Exception as e:
            # En cas d'erreur lors de l'évaluation, on garde la réponse originale
            pass

        return (answer, {'input': 0, 'output': 0})

    def _aggregate_environmental_impacts(self, impacts: List, energies: List) -> None:
        """
        Agrège les impacts environnementaux (méthode stub pour compatibilité).
        """
        # Cette méthode peut être étendue pour agréger les impacts réels
        pass

    def _format_chunks(self, chunks: List[Chunk]) -> str:
        """Formate les chunks pour l'affichage."""
        if not chunks:
            return "Aucun contexte disponible."

        return '\n\n'.join([
            f'[Doc {i + 1}] {chunk.text[:200]}...'
            for (i, chunk) in enumerate(chunks[:5])  # Limiter à 5 chunks pour l'affichage
        ])

    def _deduplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Déduplique les chunks basé sur leur contenu.

        Stratégie :
        1. Éliminer les doublons exacts (même texte)
        2. Éliminer les quasi-doublons (texte très similaire)
        3. Garder l'ordre de pertinence (les chunks les plus pertinents d'abord)
        """
        if not chunks:
            return chunks

        seen_hashes = set()
        unique_chunks = []

        for chunk in chunks:
            # Créer un hash du texte normalisé
            text_normalized = chunk.text.strip().lower()
            text_hash = hash(text_normalized)

            # Si c'est un doublon exact, on skip
            if text_hash in seen_hashes:
                continue

            # Vérifier les quasi-doublons (textes très similaires)
            is_duplicate = False
            for existing_chunk in unique_chunks[-5:]:  # Vérifier seulement les 5 derniers
                existing_text = existing_chunk.text.strip().lower()
                # Utiliser le seuil de similarité de la configuration
                if self._text_similarity(text_normalized, existing_text) > AgenticRouterConfig.CHUNK_SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_chunks.append(chunk)
                seen_hashes.add(text_hash)

        return unique_chunks

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité entre deux textes.
        Utilise une approche simple basée sur les mots communs.
        """
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _format_chunks(self, chunks: List[Chunk]) -> str:
        return '\n\n'.join([f'[Doc {i + 1}] {chunk.text[:200]}...' for (i, chunk) in enumerate(chunks)])