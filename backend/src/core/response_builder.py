from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np
from database.rag_classes import Chunk

@dataclass
class RAGResponse:
    answer: str
    nb_input_tokens: int
    nb_output_tokens: int
    context: List[Chunk]
    original_query: str
    impacts: List[Union[float, str]] = field(default_factory=lambda : [0, 0, ''])
    energy: List[Union[float, str]] = field(default_factory=lambda : [0, 0, ''])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {'answer': self.answer, 'nb_input_tokens': self.nb_input_tokens, 'nb_output_tokens': self.nb_output_tokens, 'context': self.context, 'impacts': self.impacts, 'energy': self.energy, 'original_query': self.original_query, **({'metadata': self.metadata} if self.metadata else {})}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGResponse':
        return cls(answer=data.get('answer', ''), nb_input_tokens=data.get('nb_input_tokens', 0), nb_output_tokens=data.get('nb_output_tokens', 0), context=data.get('context', []), original_query=data.get('original_query', ''), impacts=data.get('impacts', [0, 0, '']), energy=data.get('energy', [0, 0, '']), metadata=data.get('metadata', {}))

    @property
    def total_tokens(self) -> int:
        return self.nb_input_tokens + self.nb_output_tokens

class RAGResponseBuilder:

    def __init__(self):
        self._answer: str = ''
        self._nb_input_tokens: int = 0
        self._nb_output_tokens: int = 0
        self._context: List[Chunk] = []
        self._impacts: List[Union[float, str]] = [0, 0, '']
        self._energy: List[Union[float, str]] = [0, 0, '']
        self._original_query: str = ''
        self._metadata: Dict[str, Any] = {}

    def set_answer(self, answer: str) -> 'RAGResponseBuilder':
        self._answer = answer
        return self

    def set_tokens(self, input_tokens: Union[int, List[int], np.ndarray], output_tokens: Union[int, List[int], np.ndarray]) -> 'RAGResponseBuilder':
        if isinstance(input_tokens, (list, np.ndarray)):
            self._nb_input_tokens = int(np.sum(input_tokens))
        else:
            self._nb_input_tokens = int(input_tokens)
        if isinstance(output_tokens, (list, np.ndarray)):
            self._nb_output_tokens = int(np.sum(output_tokens))
        else:
            self._nb_output_tokens = int(output_tokens)
        return self

    def add_tokens(self, input_tokens: Union[int, List[int], np.ndarray], output_tokens: Union[int, List[int], np.ndarray]) -> 'RAGResponseBuilder':
        if isinstance(input_tokens, (list, np.ndarray)):
            self._nb_input_tokens += int(np.sum(input_tokens))
        else:
            self._nb_input_tokens += int(input_tokens)
        if isinstance(output_tokens, (list, np.ndarray)):
            self._nb_output_tokens += int(np.sum(output_tokens))
        else:
            self._nb_output_tokens += int(output_tokens)
        return self

    def aggregate_llm_response(self, response: Dict[str, Any]) -> 'RAGResponseBuilder':
        if response is None:
            return self
        input_t = response.get('nb_input_tokens', 0)
        output_t = response.get('nb_output_tokens', 0)
        self.add_tokens(input_t, output_t)
        if 'impacts' in response and response['impacts']:
            current_impacts = self._impacts
            response_impacts = response['impacts']
            self._impacts = [current_impacts[0] + response_impacts[0], current_impacts[1] + response_impacts[1], response_impacts[2]]
        if 'energy' in response and response['energy']:
            current_energy = self._energy
            response_energy = response['energy']
            self._energy = [current_energy[0] + response_energy[0], current_energy[1] + response_energy[1], response_energy[2]]
        return self

    def set_context(self, context: List[Chunk]) -> 'RAGResponseBuilder':
        self._context = context
        return self

    def set_query(self, query: str) -> 'RAGResponseBuilder':
        self._original_query = query
        return self

    def set_impacts(self, impacts: List[Union[float, str]]) -> 'RAGResponseBuilder':
        self._impacts = impacts
        return self

    def set_energy(self, energy: List[Union[float, str]]) -> 'RAGResponseBuilder':
        self._energy = energy
        return self

    def add_metadata(self, key: str, value: Any) -> 'RAGResponseBuilder':
        self._metadata[key] = value
        return self

    def set_metadata(self, metadata: Dict[str, Any]) -> 'RAGResponseBuilder':
        self._metadata = metadata
        return self

    def build(self) -> RAGResponse:
        if not self._answer and self._original_query:
            pass
        return RAGResponse(answer=self._answer, nb_input_tokens=self._nb_input_tokens, nb_output_tokens=self._nb_output_tokens, context=self._context, original_query=self._original_query, impacts=self._impacts, energy=self._energy, metadata=self._metadata)

    def build_dict(self) -> Dict[str, Any]:
        return self.build().to_dict()

    def reset(self) -> 'RAGResponseBuilder':
        self._answer = ''
        self._nb_input_tokens = 0
        self._nb_output_tokens = 0
        self._context = []
        self._impacts = [0, 0, '']
        self._energy = [0, 0, '']
        self._original_query = ''
        self._metadata = {}
        return self

def create_rag_response(answer: str, input_tokens: int, output_tokens: int, context: List[Chunk], query: str, impacts: Optional[List[Union[float, str]]]=None, energy: Optional[List[Union[float, str]]]=None, metadata: Optional[Dict[str, Any]]=None) -> RAGResponse:
    return RAGResponse(answer=answer, nb_input_tokens=input_tokens, nb_output_tokens=output_tokens, context=context, original_query=query, impacts=impacts or [0, 0, ''], energy=energy or [0, 0, ''], metadata=metadata or {})