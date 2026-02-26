from functools import wraps
from typing import Callable, Any, Optional, Type
import logging
from logging.handlers import RotatingFileHandler
import traceback
import os

class RAGError(Exception):

    def __init__(self, message: str, details: Optional[dict]=None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f'{self.message} - Details: {self.details}'
        return self.message

    def to_dict(self) -> dict:
        return {'error_type': self.__class__.__name__, 'message': self.message, 'details': self.details}

class ConfigurationError(RAGError):

    def __init__(self, message: str, config_key: Optional[str]=None, config_value: Optional[Any]=None):
        details = {}
        if config_key:
            details['config_key'] = config_key
        if config_value is not None:
            details['config_value'] = str(config_value)
        super().__init__(message, details)

class LLMError(RAGError):

    def __init__(self, message: str, provider: Optional[str]=None, model: Optional[str]=None, original_error: Optional[Exception]=None):
        details = {}
        if provider:
            details['provider'] = provider
        if model:
            details['model'] = model
        if original_error:
            details['original_error'] = str(original_error)
            details['original_error_type'] = type(original_error).__name__
        super().__init__(message, details)

class VectorStoreError(RAGError):

    def __init__(self, message: str, backend: Optional[str]=None, operation: Optional[str]=None, original_error: Optional[Exception]=None):
        details = {}
        if backend:
            details['backend'] = backend
        if operation:
            details['operation'] = operation
        if original_error:
            details['original_error'] = str(original_error)
            details['original_error_type'] = type(original_error).__name__
        super().__init__(message, details)

class DatabaseError(RAGError):

    def __init__(self, message: str, database: Optional[str]=None, operation: Optional[str]=None, original_error: Optional[Exception]=None):
        details = {}
        if database:
            details['database'] = database
        if operation:
            details['operation'] = operation
        if original_error:
            details['original_error'] = str(original_error)
            details['original_error_type'] = type(original_error).__name__
        super().__init__(message, details)

class IndexationError(RAGError):

    def __init__(self, message: str, document: Optional[str]=None, stage: Optional[str]=None, original_error: Optional[Exception]=None):
        details = {}
        if document:
            details['document'] = document
        if stage:
            details['stage'] = stage
        if original_error:
            details['original_error'] = str(original_error)
            details['original_error_type'] = type(original_error).__name__
        super().__init__(message, details)

class RetrievalError(RAGError):

    def __init__(self, message: str, query: Optional[str]=None, nb_chunks: Optional[int]=None, original_error: Optional[Exception]=None):
        details = {}
        if query:
            details['query'] = query[:100] + '...' if len(query) > 100 else query
        if nb_chunks is not None:
            details['nb_chunks'] = nb_chunks
        if original_error:
            details['original_error'] = str(original_error)
            details['original_error_type'] = type(original_error).__name__
        super().__init__(message, details)

def handle_errors(reraise: bool=False, default_return: Any=None, log_level: int=logging.ERROR, exception_types: Optional[tuple]=None) -> Callable:

    def decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if exception_types and (not isinstance(e, exception_types)):
                    raise
                logger = logging.getLogger(func.__module__)
                log_message = f'Error in {func.__name__}: {str(e)}'
                logger.log(log_level, log_message, exc_info=True)
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator

def safe_execute(func: Callable, *args, default_return: Any=None, error_callback: Optional[Callable[[Exception], Any]]=None, **kwargs) -> Any:
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger = logging.getLogger(func.__module__)
        logger.error(f'Error in {func.__name__}: {str(e)}', exc_info=True)
        if error_callback:
            return error_callback(e)
        return default_return

def validate_config(config: dict, required_keys: list, config_name: str='Configuration') -> None:
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ConfigurationError(f'{config_name} is missing required keys', config_key=', '.join(missing_keys), details={'missing_keys': missing_keys})

def configure_logging(level: int=logging.INFO, log_file: Optional[str]=None, format_string: Optional[str]=None, max_bytes: int=10*1024*1024, backup_count: int=5) -> None:
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'))
    logging.basicConfig(level=level, format=format_string, handlers=handlers)