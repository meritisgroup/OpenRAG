import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException

from api.schemas.storage import StorageFile, StorageListResponse, StorageDeleteResponse

router = APIRouter()

STORAGE_PATH = 'storage'


@router.get("", response_model=StorageListResponse)
def list_storage_files(prefix: Optional[str] = None):
    if not os.path.exists(STORAGE_PATH):
        return StorageListResponse(files=[])
    
    files = []
    for filename in os.listdir(STORAGE_PATH):
        if filename.endswith('.db'):
            if prefix and not filename.startswith(prefix):
                continue
            file_path = os.path.join(STORAGE_PATH, filename)
            size = os.path.getsize(file_path)
            files.append(StorageFile(filename=filename, size_bytes=size))
    
    return StorageListResponse(files=files)


@router.delete("/{filename}")
def delete_storage_file(filename: str):
    file_path = os.path.join(STORAGE_PATH, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Storage file not found")
    
    os.remove(file_path)
    return {"status": "deleted", "filename": filename}


@router.delete("/by-prefix/{prefix}", response_model=StorageDeleteResponse)
def delete_storage_by_prefix(prefix: str):
    if not os.path.exists(STORAGE_PATH):
        return StorageDeleteResponse(status="deleted", deleted_count=0, deleted_files=[])
    
    deleted_files = []
    for filename in os.listdir(STORAGE_PATH):
        if filename.startswith(prefix) and filename.endswith('.db'):
            file_path = os.path.join(STORAGE_PATH, filename)
            os.remove(file_path)
            deleted_files.append(filename)
    
    return StorageDeleteResponse(
        status="deleted",
        deleted_count=len(deleted_files),
        deleted_files=deleted_files
    )
