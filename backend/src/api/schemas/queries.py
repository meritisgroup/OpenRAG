from typing import List, Optional
from pydantic import BaseModel


class QueryFile(BaseModel):
    filename: str
    query_count: int


class QueryFileListResponse(BaseModel):
    queries: List[QueryFile]


class QueryFileContent(BaseModel):
    filename: str
    queries: List[dict]


class QueryUploadResponse(BaseModel):
    filename: str
    status: str
    query_count: int
