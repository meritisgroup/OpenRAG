from typing import Dict, List, Any, Optional, Union
import concurrent.futures
import cohere
from pydantic import BaseModel
import numpy as np
from core.interfaces.llm_provider import LLMProvider
from core.error_handler import LLMError
from utils.threading_utils import get_executor_threads
from utils.ecologits_init import init_ecologits


class CohereProvider(LLMProvider):

    def __init__(
        self,
        models_infos: Dict[str, Any],
        language: str = 'EN',
        max_attempts: int = 5,
        max_workers: int = 10
    ):
        self.models_infos = models_infos
        self.language = language
        self.max_attempts = max_attempts
        self.max_workers = max_workers
        self.temperature = 0.0
        init_ecologits()
        self.clients = self._create_clients()

    def _create_clients(self) -> Dict[str, str]:
        clients = {}
        for key in self.models_infos.keys():
            model_config = self.models_infos[key]
            provider = model_config.get('provider', '').lower()
            if provider == 'cohere':
                api_key = model_config.get('api_key', '')
                if api_key:
                    clients[key] = api_key
        return clients

    def _get_client(self, api_key: str) -> cohere.ClientV2:
        return cohere.ClientV2(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return 'cohere'

    @property
    def is_available(self) -> bool:
        return len(self.clients) > 0

    def predict(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float = 0,
        options_generation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            if options_generation is not None and options_generation.get('type_generation') == 'no_generation':
                return {
                    'texts': '',
                    'nb_input_tokens': 0,
                    'nb_output_tokens': 0,
                    'impacts': [0, 0, ''],
                    'energy': [0, 0, '']
                }

            if model not in self.clients:
                raise LLMError(
                    f'Model {model} not configured for Cohere',
                    provider='cohere',
                    model=model
                )

            api_key = self.clients[model]
            client = self._get_client(api_key)

            response = client.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=temperature or self.temperature
            )

            answer = response.message.content[0].text
            input_tokens = response.usage.tokens.input_tokens
            output_tokens = response.usage.tokens.output_tokens

            impacts = [0, 0, '']
            energy = [0, 0, '']

            try:
                if hasattr(response, 'impacts') and response.impacts:
                    impacts = [
                        response.impacts.gwp.value.min,
                        response.impacts.gwp.value.max,
                        response.impacts.gwp.unit
                    ]
                    energy = [
                        response.impacts.energy.value.min,
                        response.impacts.energy.value.max,
                        response.impacts.energy.unit
                    ]
            except Exception:
                pass

            return {
                'texts': answer,
                'nb_input_tokens': input_tokens,
                'nb_output_tokens': output_tokens,
                'impacts': impacts,
                'energy': energy
            }
        except cohere.errors.CohereError as e:
            raise LLMError(
                f'Cohere API error: {str(e)}',
                provider='cohere',
                model=model,
                original_error=e
            )
        except Exception as e:
            raise LLMError(
                f'Prediction failed for model {model}',
                provider='cohere',
                model=model,
                original_error=e
            )

    def multiple_predict(
        self,
        prompts: List[str],
        system_prompt: str,
        model: str,
        temperature: float = 0,
        max_workers: int = 10
    ) -> Dict[str, Any]:
        try:
            if max_workers <= get_executor_threads():
                max_workers = 1

            all_answers = []
            total_input_tokens = 0
            total_output_tokens = 0
            total_impacts = [0, 0, '']
            total_energy = [0, 0, '']

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(self.predict, prompt, system_prompt, model, temperature): i
                    for i, prompt in enumerate(prompts)
                }
                results = [None] * len(prompts)

                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as exc:
                        results[index] = {
                            'texts': f'ERROR: {exc}',
                            'nb_input_tokens': 0,
                            'nb_output_tokens': 0,
                            'impacts': [0, 0, ''],
                            'energy': [0, 0, '']
                        }

            for result in results:
                if result:
                    all_answers.append(result['texts'])
                    total_input_tokens += result['nb_input_tokens']
                    total_output_tokens += result['nb_output_tokens']
                    total_impacts[0] += result['impacts'][0]
                    total_impacts[1] += result['impacts'][1]
                    total_energy[0] += result['energy'][0]
                    total_energy[1] += result['energy'][1]

            return {
                'texts': all_answers,
                'nb_input_tokens': total_input_tokens,
                'nb_output_tokens': total_output_tokens,
                'impacts': total_impacts,
                'energy': total_energy
            }
        except Exception as e:
            raise LLMError(
                f'Multiple predictions failed for model {model}',
                provider='cohere',
                model=model,
                original_error=e
            )

    def predict_json(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        json_format: type[BaseModel],
        temperature: float = 0
    ) -> Optional[BaseModel]:
        try:
            if model not in self.clients:
                raise LLMError(
                    f'Model {model} not configured for Cohere',
                    provider='cohere',
                    model=model
                )

            api_key = self.clients[model]
            client = self._get_client(api_key)

            response = client.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                response_format={"type": "json_object"},
                temperature=temperature or self.temperature
            )

            import json
            answer = response.message.content[0].text
            parsed = json.loads(answer)
            return json_format.model_validate(parsed)
        except Exception as e:
            raise LLMError(
                f'JSON prediction failed for model {model}',
                provider='cohere',
                model=model,
                original_error=e
            )

    def embeddings(self, texts: Union[str, List[str]], model: str, input_type: Optional[str] = None) -> Dict[str, Any]:
        try:
            if model not in self.clients:
                raise LLMError(
                    f'Model {model} not configured for Cohere',
                    provider='cohere',
                    model=model
                )

            api_key = self.clients[model]
            client = self._get_client(api_key)

            if isinstance(texts, str):
                texts = [texts]

            response = client.embed(
                texts=texts,
                model=model,
                input_type=input_type or "search_document",
                embedding_types=["float"]
            )

            embeddings = [item.embedding for item in response.embeddings]

            return {
                'embeddings': embeddings,
                'model': model,
                'nb_tokens': response.meta.tokens.input_tokens + response.meta.tokens.output_tokens,
                'usage': {
                    'prompt_tokens': response.meta.tokens.input_tokens,
                    'total_tokens': response.meta.tokens.input_tokens + response.meta.tokens.output_tokens
                }
            }
        except cohere.errors.CohereError as e:
            raise LLMError(
                f'Cohere API error: {str(e)}',
                provider='cohere',
                model=model,
                original_error=e
            )
        except Exception as e:
            raise LLMError(
                f'Embeddings failed for model {model}',
                provider='cohere',
                model=model,
                original_error=e
            )

    def predict_image(
        self,
        prompt: str,
        model: str,
        image: np.ndarray,
        json_format: Optional[type[BaseModel]] = None,
        temperature: float = 0
    ) -> Dict[str, Any]:
        raise LLMError(
            'Cohere does not provide native image analysis. Please configure a vision model (e.g., Anthropic).',
            provider='cohere',
            model=model
        )

    def predict_images(
        self,
        prompts: List[str],
        model: str,
        images: List[np.ndarray],
        json_format: Optional[type[BaseModel]] = None,
        temperature: float = 0,
        max_workers: int = 10
    ) -> List[Dict[str, Any]]:
        raise LLMError(
            'Cohere does not provide native image analysis. Please configure a vision model (e.g., Anthropic).',
            provider='cohere',
            model=model
        )

    def reranking(
        self,
        query: str,
        chunk_list: List,
        model: str,
        max_workers: int = 10
    ) -> Dict[str, Any]:
        try:
            if model not in self.clients:
                raise LLMError(
                    f'Model {model} not configured for Cohere',
                    provider='cohere',
                    model=model
                )

            api_key = self.clients[model]
            client = self._get_client(api_key)

            documents = [chunk.text for chunk in chunk_list]

            response = client.rerank(
                query=query,
                documents=documents,
                model=model
            )

            ordered_by_index = sorted(response.results, key=lambda x: x.index)
            scores = [item.relevance_score for item in ordered_by_index]

            input_tokens = response.meta.tokens.input_tokens + response.meta.tokens.output_tokens

            return {
                'scores': scores,
                'nb_input_tokens': [input_tokens]
            }
        except cohere.errors.CohereError as e:
            raise LLMError(
                f'Cohere API error: {str(e)}',
                provider='cohere',
                model=model,
                original_error=e
            )
        except Exception as e:
            raise LLMError(
                f'Reranking failed for model {model}',
                provider='cohere',
                model=model,
                original_error=e
            )

    def release_memory(self) -> None:
        pass
