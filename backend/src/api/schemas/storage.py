from typing import List
from pydantic import BaseModel


class StorageFile(BaseModel):
    filename: str
    size_bytes: int


class StorageListResponse(BaseModel):
    files: List[StorageFile]


class StorageDeleteResponse(BaseModel):
    status: str
    deleted_count: int
    deleted_files: List[str]
