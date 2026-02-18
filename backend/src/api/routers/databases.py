import os
import json
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from elasticsearch import Elasticsearch

from api.schemas.database import (
    DatabaseInfo, DatabaseCreateRequest, DatabaseListResponse,
    MetadatasResponse, MetadatasUpdateRequest, DatabaseDocumentUploadResponse
)
from database.utils import get_list_path_documents

router = APIRouter()

DATABASES_PATH = 'data/databases'
STORAGE_PATH = 'storage'


def _get_elasticsearch_client():
    config_path = 'data/base_config_server.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    params = config.get('params_vectorbase', {})
    return Elasticsearch(
        [params.get('url', 'http://localhost:9200')],
        basic_auth=(params.get('auth', ['elastic', ''])[0], params.get('auth', ['', ''])[1]),
        verify_certs=False,
        ssl_show_warn=False
    )


@router.get("", response_model=DatabaseListResponse)
def list_databases():
    databases = []
    if os.path.exists(DATABASES_PATH):
        for db_name in os.listdir(DATABASES_PATH):
            db_path = os.path.join(DATABASES_PATH, db_name)
            if os.path.isdir(db_path) and db_name != '.gitkeep':
                doc_count = 0
                docs_path = os.path.join(db_path, 'documents')
                if os.path.exists(docs_path):
                    doc_count = len([f for f in os.listdir(docs_path) if os.path.isfile(os.path.join(docs_path, f))])
                
                databases.append(DatabaseInfo(
                    name=db_name,
                    path=db_path,
                    document_count=doc_count,
                    created_at=None
                ))
    
    return DatabaseListResponse(databases=databases)


@router.post("")
def create_database(request: DatabaseCreateRequest):
    db_path = os.path.join(DATABASES_PATH, request.name)
    if os.path.exists(db_path):
        raise HTTPException(status_code=400, detail="Database already exists")
    
    os.makedirs(db_path, exist_ok=True)
    os.makedirs(os.path.join(db_path, 'documents'), exist_ok=True)
    
    return {
        "status": "created",
        "name": request.name,
        "path": db_path
    }


@router.delete("/{name}")
def delete_database(name: str):
    db_path = os.path.join(DATABASES_PATH, name)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")
    
    shutil.rmtree(db_path)
    
    try:
        es = _get_elasticsearch_client()
        for index_name in es.indices.get_alias(index='*').keys():
            if name in index_name:
                es.indices.delete(index=index_name)
    except Exception as e:
        pass
    
    if os.path.exists(STORAGE_PATH):
        for storage_file in os.listdir(STORAGE_PATH):
            if name in storage_file:
                os.remove(os.path.join(STORAGE_PATH, storage_file))
    
    return {"status": "deleted", "name": name}


@router.get("/{name}", response_model=DatabaseInfo)
def get_database(name: str):
    db_path = os.path.join(DATABASES_PATH, name)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")
    
    doc_count = 0
    docs_path = os.path.join(db_path, 'documents')
    if os.path.exists(docs_path):
        doc_count = len([f for f in os.listdir(docs_path) if os.path.isfile(os.path.join(docs_path, f))])
    
    return DatabaseInfo(
        name=name,
        path=db_path,
        document_count=doc_count,
        created_at=None
    )


@router.get("/{name}/documents")
def list_database_documents(name: str):
    db_path = os.path.join(DATABASES_PATH, name)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")
    
    try:
        documents = get_list_path_documents(db_path)
        return {"database": name, "documents": documents}
    except Exception as e:
        return {"database": name, "documents": [], "error": str(e)}


@router.post("/{name}/documents/upload", response_model=DatabaseDocumentUploadResponse)
async def upload_database_document(name: str, file: UploadFile = File(...)):
    db_path = os.path.join(DATABASES_PATH, name)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")
    
    filename = file.filename or "unnamed"
    
    docs_path = os.path.join(db_path, 'documents')
    os.makedirs(docs_path, exist_ok=True)
    
    file_path = os.path.join(docs_path, filename)
    content = await file.read()
    
    with open(file_path, 'wb') as f:
        f.write(content)
    
    return DatabaseDocumentUploadResponse(
        database_name=name,
        document_name=filename,
        status="uploaded",
        path=file_path
    )


@router.get("/{name}/metadatas", response_model=MetadatasResponse)
def get_database_metadatas(name: str):
    db_path = os.path.join(DATABASES_PATH, name)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")
    
    metadatas_path = os.path.join(db_path, 'metadatas.json')
    
    if not os.path.exists(metadatas_path):
        return MetadatasResponse(database_name=name, metadatas={})
    
    with open(metadatas_path, 'r', encoding='utf-8') as f:
        metadatas = json.load(f)
    
    return MetadatasResponse(database_name=name, metadatas=metadatas)


@router.put("/{name}/metadatas", response_model=MetadatasResponse)
def update_database_metadatas(name: str, request: MetadatasUpdateRequest):
    db_path = os.path.join(DATABASES_PATH, name)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")
    
    metadatas_path = os.path.join(db_path, 'metadatas.json')
    
    with open(metadatas_path, 'w', encoding='utf-8') as f:
        json.dump(request.metadatas, f, indent=4, ensure_ascii=False)
    
    return MetadatasResponse(database_name=name, metadatas=request.metadatas)


@router.delete("/{name}/documents/{document_name}")
def delete_database_document(name: str, document_name: str):
    db_path = os.path.join(DATABASES_PATH, name)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")
    
    doc_path = os.path.join(db_path, 'documents', document_name)
    if not os.path.exists(doc_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    os.remove(doc_path)
    return {"status": "deleted", "database": name, "document": document_name}
