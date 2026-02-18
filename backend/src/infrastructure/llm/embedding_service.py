from typing import Dict, List, Any, Union
import numpy as np
from core.interfaces.illm_provider import ILLMProvider
from core.error_handler import LLMError

class EmbeddingService:

    def __init__(self, provider: ILLMProvider):
        self.provider = provider

    def embed_text(self, text: str, model: str) -> List[float]:
        result = self.provider.embeddings(texts=text, model=model)
        return result['embeddings'][0]

    def embed_texts(self, texts: List[str], model: str) -> List[List[float]]:
        result = self.provider.embeddings(texts=texts, model=model)
        return result['embeddings']

    def embed_with_usage(self, texts: Union[str, List[str]], model: str) -> Dict[str, Any]:
        return self.provider.embeddings(texts=texts, model=model)

    def cosine_similarity(self, embedding1: Union[List[float], np.ndarray], embedding2: Union[List[float], np.ndarray]) -> float:
        a = np.array(embedding1)
        b = np.array(embedding2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def batch_embed(self, texts: List[str], model: str, batch_size: int=100) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_texts(batch, model)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings