from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import json
from methods.advanced_rag.indexation import AdvancedIndexation
from methods.naive_rag.indexation import contexts_to_prompts
from methods.advanced_rag.agent import AdvancedRag
from methods.naive_rag.query import NaiveSearch
from methods.advanced_rag.reranker import Reranker
from methods.query_reformulation.query_reformulation import query_reformulation
from .agents_router import RouterAgent, RetrieverAgent, ReasonerAgent, SynthesizerAgent, EvaluatorAgent
from .abstract_class import QueryPlan
from database.rag_classes import Chunk

class AgenticRouterRAG(AdvancedRag):

    def __init__(self, config_server: dict, models_infos: dict, dbs_name: list[str], data_folders_name: list[str]) -> None:
        super().__init__(config_server=config_server, models_infos=models_infos, dbs_name=dbs_name, data_folders_name=data_folders_name)
        self.max_iterations = 3
        self.enable_self_correction = True
        self.confidence_threshold = 0.7
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
        (router_response, input_tokens, output_tokens) = self.router_agent.process(query)
        query_plan: QueryPlan = router_response.content
        execution_trace.append({'step': 'routing', 'response': router_response, 'plan': query_plan})
        all_chunks = []
        sub_responses = []
        query_plan.sub_queries.append(query)
        for (i, sub_query) in enumerate(query_plan.sub_queries, 1):
            retrieval_response = self.retriever_agent.process({'query': sub_query, 'strategy': query_plan.retrieval_strategy, 'nb_chunks': query_plan.nb_chunks_per_query})
            chunks = retrieval_response.content
            all_chunks.extend(chunks)
            if query_plan.requires_reasoning:
                (reasoning_response, input_tokens, output_tokens) = self.reasoner_agent.process({'query': sub_query, 'chunks': chunks})
                sub_responses.append({'query': sub_query, 'content': reasoning_response.content, 'confidence': reasoning_response.confidence, 'chunks': chunks})
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            else:
                sub_responses.append({'query': sub_query, 'chunks': chunks, 'confidence': retrieval_response.confidence})
        execution_trace.append({'step': 'retrieval', 'nb_chunks': len(all_chunks), 'sub_responses': sub_responses})
        if query_plan.requires_synthesis and len(sub_responses) > 1:
            (synthesis_response, input_tokens, output_tokens) = self.synthesizer_agent.process({'query': query, 'sub_responses': sub_responses})
            answer = synthesis_response.content
            confidence = synthesis_response.confidence
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
        elif query_plan.requires_reasoning:
            if len(sub_responses) > 0 and 'content' in sub_responses[0]:
                answer = sub_responses[0]['content']
                confidence = sub_responses[0]['confidence']
            else:
                (reasoning_response, input_tokens, output_tokens) = self.reasoner_agent.process({'query': query, 'chunks': all_chunks})
                answer = reasoning_response.content
                confidence = reasoning_response.confidence
        else:
            prompt = self.build_final_prompt(all_chunks, query)
            if options_generation is None:
                options_generation = self.config_server.get('options_generation', {})
            response = self.agent.predict(prompt=prompt, system_prompt=self.system_prompt, options_generation=options_generation, model=self.llm_model)
            answer = response['texts']
            confidence = 0.75
            total_input_tokens += response.get('nb_input_tokens', 0)
            total_output_tokens += response.get('nb_output_tokens', 0)
        execution_trace.append({'step': 'generation', 'answer': answer, 'confidence': confidence})
        if self.enable_self_correction:
            (eval_response, input_tokens, output_tokens) = self.evaluator_agent.process({'query': query, 'answer': answer, 'chunks': all_chunks})
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            evaluation = eval_response.content
            execution_trace.append({'step': 'evaluation', 'evaluation': evaluation})
            if evaluation.get('needs_improvement', False) and evaluation.get('score_global', 1.0) < query_plan.confidence_threshold and (iteration < self.max_iterations):
                correction_prompt = self.prompts['correction']['QUERY_TEMPLATE'].format(query=query, answer=answer, evaluation=evaluation.get('suggestions', 'Qualité insuffisante'), context=self._format_chunks(all_chunks))
                system_prompt = self.prompts['correction']['SYSTEM_PROMPT']
                corrected_response = self.agent.predict(prompt=correction_prompt, system_prompt=system_prompt, model=self.llm_model)
                total_input_tokens += corrected_response.get('nb_input_tokens', 0)
                total_output_tokens += corrected_response.get('nb_output_tokens', 0)
                impacts[2] = corrected_response['impacts'][2]
                impacts[0] += corrected_response['impacts'][0]
                impacts[1] += corrected_response['impacts'][1]
                energies[2] = corrected_response['energy'][2]
                energies[0] += corrected_response['energy'][0]
                energies[1] += corrected_response['energy'][1]
                answer = corrected_response['texts']
                execution_trace.append({'step': 'correction', 'improved_answer': answer})
        result = {'answer': answer, 'confidence': confidence, 'nb_input_tokens': total_input_tokens, 'nb_output_tokens': total_output_tokens, 'context': all_chunks, 'impacts': impacts, 'energy': energies, 'query_plan': {'type': query_plan.query_type, 'sub_queries': query_plan.sub_queries, 'strategy': query_plan.retrieval_strategy}, 'metadata': {'nb_chunks_retrieved': len(all_chunks), 'nb_sub_queries': len(query_plan.sub_queries), 'requires_reasoning': query_plan.requires_reasoning, 'requires_synthesis': query_plan.requires_synthesis}}
        return result

    def _format_chunks(self, chunks: List[Chunk]) -> str:
        return '\n\n'.join([f'[Doc {i + 1}] {chunk.text[:200]}...' for (i, chunk) in enumerate(chunks)])