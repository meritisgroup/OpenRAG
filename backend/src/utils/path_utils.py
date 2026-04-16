import os
from pathlib import Path
from fastapi import HTTPException


def safe_join(base_dir: str, user_path: str) -> str:
    resolved = Path(os.path.join(base_dir, user_path)).resolve()
    base_resolved = Path(base_dir).resolve()
    if not str(resolved).startswith(str(base_resolved)):
        raise HTTPException(status_code=400, detail="Invalid path")
    return str(resolved)
