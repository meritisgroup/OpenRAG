from typing import Dict, List, Any, Optional, Union
from ecologits import EcoLogits

_ECOLOGITS_INITIALIZED = False


def init_ecologits() -> None:
    global _ECOLOGITS_INITIALIZED
    if not _ECOLOGITS_INITIALIZED:
        EcoLogits.init()
        _ECOLOGITS_INITIALIZED = True


def is_ecologits_initialized() -> bool:
    return _ECOLOGITS_INITIALIZED
