from typing import List, Tuple, Dict, Any
import numpy as np
from core.interfaces.illm_provider import ILLMProvider
from database.rag_classes import Chunk
from core.error_handler import RAGError

class RerankerService:

    def __init__(self, provider: ILLMProvider, model: str, max_workers: int=10):
        self.provider = provider
        self.model = model
        self.max_workers = max_workers

    def rerank(self, query: str, chunks: List[Chunk], max_contexts: int=None) -> Tuple[List[Chunk], Dict[str, Any]]:
        if not chunks:
            return ([], {'nb_input_tokens': 0, 'nb_output_tokens': 0, 'scores': [], 'original_indices': []})
        try:
            result = self.provider.reranking(query=query, chunk_list=chunks, model=self.model, max_workers=self.max_workers)
            scores = result['scores']
            input_tokens = result.get('nb_input_tokens', [])
            scored_chunks = list(zip(chunks, scores, range(len(chunks))))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            reranked_chunks = [chunk for (chunk, score, idx) in scored_chunks]
            if max_contexts is not None and max_contexts > 0:
                reranked_chunks = reranked_chunks[:max_contexts]
            metadata = {'nb_input_tokens': sum(input_tokens) if isinstance(input_tokens, list) else input_tokens, 'nb_output_tokens': 0, 'scores': scores, 'original_indices': [idx for (chunk, score, idx) in scored_chunks]}
            return (reranked_chunks, metadata)
        except Exception as e:
            raise RAGError(f'Reranking failed: {str(e)}', details={'query': query[:100], 'nb_chunks': len(chunks)})

    def rerank_multiple(self, queries: List[str], chunk_lists: List[List[Chunk]], max_contexts: int=None) -> Tuple[List[List[Chunk]], Dict[str, Any]]:
        all_reranked = []
        all_input_tokens = 0
        all_scores = []
        for (query, chunks) in zip(queries, chunk_lists):
            (reranked, metadata) = self.rerank(query, chunks, max_contexts)
            all_reranked.append(reranked)
            all_input_tokens += metadata['nb_input_tokens']
            all_scores.extend(metadata.get('scores', []))
        aggregated_metadata = {'nb_input_tokens': all_input_tokens, 'nb_output_tokens': 0, 'scores': all_scores}
        return (all_reranked, aggregated_metadata)

    def score_chunk(self, query: str, chunk: Chunk) -> float:
        try:
            result = self.provider.reranking(query=query, chunk_list=[chunk], model=self.model, max_workers=1)
            return result['scores'][0] if result['scores'] else 0.0
        except Exception:
            return 0.0

    @property
    def reranker_type(self) -> str:
        if hasattr(self.provider, 'models_infos') and self.model in self.provider.models_infos:
            return self.provider.models_infos[self.model].get('type', 'llm')
        return 'llm'

    @property
    def requires_tokens(self) -> bool:
        reranker_type = self.reranker_type
        return reranker_type in ['llm', 'embedding']