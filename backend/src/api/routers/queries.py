import os
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd

from api.schemas.queries import (
    QueryFile, QueryFileListResponse, QueryFileContent, QueryUploadResponse
)

router = APIRouter()

QUERIES_PATH = 'data/queries'


@router.get("", response_model=QueryFileListResponse)
def list_query_files():
    if not os.path.exists(QUERIES_PATH):
        return QueryFileListResponse(queries=[])
    
    queries = []
    for filename in os.listdir(QUERIES_PATH):
        if filename.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
            file_path = os.path.join(QUERIES_PATH, filename)
            try:
                df = pd.read_excel(file_path, index_col=0)
                queries.append(QueryFile(
                    filename=filename,
                    query_count=len(df)
                ))
            except Exception:
                queries.append(QueryFile(
                    filename=filename,
                    query_count=0
                ))
    
    return QueryFileListResponse(queries=queries)


@router.post("/upload", response_model=QueryUploadResponse)
async def upload_query_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
        raise HTTPException(status_code=400, detail="Only Excel files are supported")
    
    os.makedirs(QUERIES_PATH, exist_ok=True)
    
    file_path = os.path.join(QUERIES_PATH, file.filename)
    content = await file.read()
    
    with open(file_path, 'wb') as f:
        f.write(content)
    
    try:
        df = pd.read_excel(file_path, index_col=0)
        query_count = len(df)
    except Exception:
        query_count = 0
    
    return QueryUploadResponse(
        filename=file.filename,
        status="uploaded",
        query_count=query_count
    )


@router.get("/{filename}", response_model=QueryFileContent)
def get_query_file(filename: str):
    file_path = os.path.join(QUERIES_PATH, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Query file not found")
    
    try:
        df = pd.read_excel(file_path, index_col=0, dtype={'query': str, 'answer': str})
        queries = df.reset_index().to_dict(orient='records')
        return QueryFileContent(filename=filename, queries=queries)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@router.delete("/{filename}")
def delete_query_file(filename: str):
    file_path = os.path.join(QUERIES_PATH, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Query file not found")
    
    os.remove(file_path)
    return {"status": "deleted", "filename": filename}
