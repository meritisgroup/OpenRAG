from typing import Dict, List, Union
import numpy as np

class TokenCounterMixin:

    def __init__(self):
        self._nb_input_tokens = 0
        self._nb_output_tokens = 0

    @property
    def nb_input_tokens(self) -> int:
        return self._nb_input_tokens

    @property
    def nb_output_tokens(self) -> int:
        return self._nb_output_tokens

    @property
    def total_tokens(self) -> int:
        return self._nb_input_tokens + self._nb_output_tokens

    def reset_tokens(self) -> None:
        self._nb_input_tokens = 0
        self._nb_output_tokens = 0

    def add_tokens(self, input_tokens: Union[int, List[int], np.ndarray], output_tokens: Union[int, List[int], np.ndarray]) -> None:
        if isinstance(input_tokens, (list, np.ndarray)):
            self._nb_input_tokens += int(np.sum(input_tokens))
        else:
            self._nb_input_tokens += int(input_tokens)
        if isinstance(output_tokens, (list, np.ndarray)):
            self._nb_output_tokens += int(np.sum(output_tokens))
        else:
            self._nb_output_tokens += int(output_tokens)

    def aggregate_response_tokens(self, response: Dict) -> None:
        if response is None:
            return
        input_tokens = response.get('nb_input_tokens', 0)
        output_tokens = response.get('nb_output_tokens', 0)
        if isinstance(input_tokens, dict):
            input_tokens = input_tokens.get('total_tokens', 0)
        if isinstance(output_tokens, dict):
            output_tokens = output_tokens.get('total_tokens', 0)
        self.add_tokens(input_tokens, output_tokens)

    def aggregate_multiple_responses(self, responses: List[Dict]) -> None:
        for response in responses:
            self.aggregate_response_tokens(response)

    def get_token_summary(self) -> Dict[str, int]:
        return {'nb_input_tokens': self._nb_input_tokens, 'nb_output_tokens': self._nb_output_tokens, 'total_tokens': self.total_tokens}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(input_tokens={self._nb_input_tokens}, output_tokens={self._nb_output_tokens})'