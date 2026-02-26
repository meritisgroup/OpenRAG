from fastapi import APIRouter, HTTPException
from api.schemas.rag import SessionCreate, SessionInfo
from api.main import create_session as create_session_id, delete_session as delete_session_by_id, get_session_info

router = APIRouter()


@router.post("", response_model=SessionInfo)
def create_session(request: SessionCreate = None):
    session_id = create_session_id()
    info = get_session_info(session_id)
    return SessionInfo(**info)


@router.delete("/{session_id}")
def delete_session(session_id: str):
    if not delete_session_by_id(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.get("/{session_id}", response_model=SessionInfo)
def get_session(session_id: str):
    info = get_session_info(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfo(**info)
