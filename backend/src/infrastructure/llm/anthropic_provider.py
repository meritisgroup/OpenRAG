from typing import Dict, List, Any, Optional, Union
import concurrent.futures
import anthropic
from pydantic import BaseModel
import numpy as np
from core.interfaces.llm_provider import LLMProvider
from core.error_handler import LLMError
from utils.threading_utils import get_executor_threads
from utils.ecologits_init import init_ecologits


class AnthropicProvider(LLMProvider):
    
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

    def _create_clients(self) -> Dict[str, anthropic.Anthropic]:
        clients = {}
        for key in self.models_infos.keys():
            model_config = self.models_infos[key]
            provider = model_config.get('provider', '').lower()
            if provider == 'anthropic':
                api_key = model_config.get('api_key', '')
                if api_key:
                    clients[key] = anthropic.Anthropic(api_key=api_key)
        return clients

    @property
    def provider_name(self) -> str:
        return 'anthropic'

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
                    f'Model {model} not configured for Anthropic',
                    provider='anthropic',
                    model=model
                )
            
            client = self.clients[model]
            
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=temperature or self.temperature
            )
            
            answer = response.content[0].text
            
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
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
        except anthropic.APIError as e:
            raise LLMError(
                f'Anthropic API error: {str(e)}',
                provider='anthropic',
                model=model,
                original_error=e
            )
        except Exception as e:
            raise LLMError(
                f'Prediction failed for model {model}',
                provider='anthropic',
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
                provider='anthropic',
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
                    f'Model {model} not configured for Anthropic',
                    provider='anthropic',
                    model=model
                )
            
            client = self.clients[model]
            
            tool_schema = {
                'name': json_format.__name__,
                'input_schema': json_format.model_json_schema()
            }
            
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{'role': 'user', 'content': prompt}],
                tools=[tool_schema],
                tool_choice={'type': 'tool', 'name': json_format.__name__}
            )
            
            for block in response.content:
                if block.type == 'tool_use':
                    return json_format.model_validate(block.input)
            
            return None
        except Exception as e:
            raise LLMError(
                f'JSON prediction failed for model {model}',
                provider='anthropic',
                model=model,
                original_error=e
            )

    def embeddings(self, texts: Union[str, List[str]], model: str, input_type: Optional[str] = None) -> Dict[str, Any]:
        raise LLMError(
            'Anthropic does not provide native embeddings. Please configure an external embedding model (e.g., OpenAI text-embedding-3-small).',
            provider='anthropic',
            model=model
        )

    def predict_image(
        self,
        prompt: str,
        model: str,
        image: np.ndarray,
        json_format: Optional[type[BaseModel]] = None,
        temperature: float = 0
    ) -> Dict[str, Any]:
        try:
            import base64
            from io import BytesIO
            from PIL import Image
            
            if model not in self.clients:
                raise LLMError(
                    f'Model {model} not configured for Anthropic',
                    provider='anthropic',
                    model=model
                )
            
            client = self.clients[model]
            
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            else:
                img = image
            
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            image_data = base64.standard_b64encode(buffer.getvalue()).decode('utf-8')
            
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/png',
                                'data': image_data
                            }
                        },
                        {
                            'type': 'text',
                            'text': prompt
                        }
                    ]
                }]
            )
            
            answer = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            if json_format:
                import json
                try:
                    parsed = json.loads(answer)
                    return json_format.model_validate(parsed)
                except Exception:
                    return None
            
            return {
                'texts': answer,
                'nb_input_tokens': input_tokens,
                'nb_output_tokens': output_tokens,
                'impacts': [1, 0, ''],
                'energy': [0, 0, '']
            }
        except Exception as e:
            raise LLMError(
                f'Image prediction failed for model {model}',
                provider='anthropic',
                model=model,
                original_error=e
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
        try:
            if max_workers <= get_executor_threads():
                max_workers = 1
            
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(
                        self.predict_image,
                        prompts[i],
                        model,
                        images[i],
                        json_format,
                        temperature
                    ): i
                    for i in range(len(images))
                }
                
                unordered_results = []
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        unordered_results.append((index, result))
                    except Exception as exc:
                        unordered_results.append((index, None))
                
                unordered_results.sort(key=lambda x: x[0])
                results = [r[1] for r in unordered_results]
            
            return results
        except Exception as e:
            raise LLMError(
                f'Multiple image predictions failed for model {model}',
                provider='anthropic',
                model=model,
                original_error=e
            )

    def reranking(
        self,
        query: str,
        chunk_list: List,
        model: str,
        max_workers: int = 10
    ) -> Dict[str, Any]:
        raise LLMError(
            'Anthropic does not provide native reranking. Please configure an external reranker model.',
            provider='anthropic',
            model=model
        )

    def release_memory(self) -> None:
        pass
