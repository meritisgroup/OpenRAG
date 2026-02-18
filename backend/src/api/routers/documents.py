import os
from fastapi import APIRouter, HTTPException, UploadFile, File

from api.schemas.database import DocumentUploadResponse
from utils.open_doc import Opener

router = APIRouter()

DOCUMENTS_PATH = 'data/documents'


@router.post("/process")
async def process_document(file: UploadFile = File(...)):
    content = await file.read()
    
    temp_path = os.path.join(DOCUMENTS_PATH, f"temp_{file.filename}")
    os.makedirs(DOCUMENTS_PATH, exist_ok=True)
    
    with open(temp_path, 'wb') as f:
        f.write(content)
    
    try:
        text = Opener(save=False).open_doc(temp_path)
        os.remove(temp_path)
        return {
            "filename": file.filename,
            "content": text.strip() if text else "",
            "is_valid": bool(text and text.strip())
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not os.path.exists(DOCUMENTS_PATH):
        os.makedirs(DOCUMENTS_PATH, exist_ok=True)
    
    file_path = os.path.join(DOCUMENTS_PATH, file.filename)
    
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)
    
    return DocumentUploadResponse(
        document_id=file.filename,
        name=file.filename,
        status="uploaded",
        chunk_count=0
    )


@router.get("/{document_id}/content")
def get_document_content(document_id: str):
    file_path = os.path.join(DOCUMENTS_PATH, document_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        content = Opener(save=False).open_doc(file_path)
        return {"document_id": document_id, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
def delete_document(document_id: str):
    file_path = os.path.join(DOCUMENTS_PATH, document_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    os.remove(file_path)
    return {"status": "deleted", "document_id": document_id}


@router.get("")
def list_documents():
    if not os.path.exists(DOCUMENTS_PATH):
        return {"documents": []}
    
    documents = [f for f in os.listdir(DOCUMENTS_PATH) if os.path.isfile(os.path.join(DOCUMENTS_PATH, f))]
    return {"documents": documents}
