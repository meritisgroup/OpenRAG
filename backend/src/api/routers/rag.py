import json
import os
import shutil
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from elasticsearch import Elasticsearch

from api.schemas.rag import (
    CreateAgentRequest, GenerateRequest, GenerateResponse,
    IndexRequest, RAGMethod, AgentStatus, ChunkInfo
)
from api.main import set_agent, session_exists, get_agent, get_session_info
from factory_RagAgent import get_rag_agent, get_custom_rag_agent, change_config_server
from factory import RAGFactory
from utils.factory_name_dataset_vectorbase import get_name

router = APIRouter()


class GenerateNamesRequest(BaseModel):
    rag_name: str
    config: Dict[str, Any]
    additional_name: str = ''


class CustomRagRequest(BaseModel):
    name: str
    config: Dict[str, Any]


class MergeRagRequest(BaseModel):
    name: str
    rag_list: List[str]
    rag_config_list: List[Dict[str, Any]]
    config: Dict[str, Any]


@router.get("/methods", response_model=list[RAGMethod])
def list_rag_methods():
    methods = []
    all_rags_path = 'data/all_rags.json'
    if os.path.exists(all_rags_path):
        with open(all_rags_path, 'r') as f:
            all_rags = json.load(f)
        for method_id, method_name in all_rags.items():
            methods.append(RAGMethod(id=method_id, name=method_name))
    else:
        for method_id in RAGFactory.list_available_rags():
            methods.append(RAGMethod(id=method_id, name=method_id))
    return methods


@router.post("/create")
def create_agent(request: CreateAgentRequest):
    if not session_exists(request.session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    custom_rags_path = f'data/custom_rags/{request.rag_method}.json'
    merge_rags_path = f'data/merge/{request.rag_method}.json'
    
    if os.path.exists(custom_rags_path):
        with open(custom_rags_path, 'r') as f:
            custom_config = json.load(f)
        custom_config['params_host_llm'] = request.config.get('params_host_llm', {})
        agent = get_rag_agent(
            rag_name=custom_config.get('base', request.rag_method),
            config_server=custom_config,
            models_infos=request.models_infos,
            databases_name=request.databases
        )
    elif os.path.exists(merge_rags_path):
        with open(merge_rags_path, 'r') as f:
            merge_config = json.load(f)
        merge_config['params_host_llm'] = request.config.get('params_host_llm', {})
        agent = get_rag_agent(
            rag_name='merger',
            config_server=merge_config,
            models_infos=request.models_infos,
            databases_name=request.databases
        )
    else:
        config = change_config_server(
            rag_name=request.rag_method,
            config_server=request.config
        )
        agent = get_rag_agent(
            rag_name=request.rag_method,
            config_server=config,
            models_infos=request.models_infos,
            databases_name=request.databases
        )
    
    set_agent(
        session_id=request.session_id,
        agent=agent,
        rag_method=request.rag_method,
        databases=request.databases
    )
    
    return {
        "status": "created",
        "session_id": request.session_id,
        "rag_method": request.rag_method
    }


@router.post("/index")
def run_indexation(request: IndexRequest):
    agent = get_agent(request.session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found for session")
    
    agent.indexation_phase(
        reset_index=request.reset_index,
        reset_preprocess=request.reset_preprocess
    )
    
    return {
        "status": "indexed",
        "session_id": request.session_id
    }


@router.post("/generate", response_model=GenerateResponse)
def generate_answer(request: GenerateRequest):
    agent = get_agent(request.session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found for session")
    
    start_time = time.time()
    
    result = agent.generate_answer(
        query=request.query,
        nb_chunks=request.nb_chunks,
        options_generation=request.options_generation
    )
    
    end_time = time.time()
    
    context = result.get('context', [])
    if isinstance(context, list):
        context_data = [
            ChunkInfo(
                text=chunk.text if hasattr(chunk, 'text') else str(chunk),
                document=getattr(chunk, 'document', None),
                rerank_score=getattr(chunk, 'rerank_score', None),
                chunk_id=getattr(chunk, 'chunk_id', None)
            ) for chunk in context
        ]
    else:
        context_data = context
    
    return GenerateResponse(
        answer=result.get('answer', ''),
        nb_input_tokens=result.get('nb_input_tokens', 0),
        nb_output_tokens=result.get('nb_output_tokens', 0),
        context=context_data,
        impacts=result.get('impacts', [0, 0, '']),
        energy=result.get('energy', [0, 0, '']),
        original_query=request.query,
        time=end_time - start_time
    )


@router.get("/status/{session_id}", response_model=AgentStatus)
def get_agent_status(session_id: str):
    info = get_session_info(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    agent = get_agent(session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found for session")
    
    return AgentStatus(
        session_id=session_id,
        rag_method=info.get('rag_method', 'unknown'),
        databases=info.get('databases', []),
        total_tokens=agent.total_tokens if hasattr(agent, 'total_tokens') else 0,
        nb_input_tokens=agent.nb_input_tokens if hasattr(agent, 'nb_input_tokens') else 0,
        nb_output_tokens=agent.nb_output_tokens if hasattr(agent, 'nb_output_tokens') else 0
    )


@router.post("/generate-names")
def generate_names(request: GenerateNamesRequest):
    names = get_name(
        rag_name=request.rag_name,
        config_server=request.config,
        additionnal_name=request.additional_name
    )
    return {"names": names}


@router.get("/elasticsearch/indices")
def list_elasticsearch_indices(prefix: Optional[str] = None):
    config_path = 'data/base_config_server.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    params = config.get('params_vectorbase', {})
    es = Elasticsearch(
        [params.get('url', 'http://localhost:9200')],
        basic_auth=(params.get('auth', ['elastic', ''])[0], params.get('auth', ['', ''])[1]),
        verify_certs=False,
        ssl_show_warn=False
    )
    
    try:
        indices = list(es.indices.get_alias(index='*').keys())
    except Exception as e:
        return {"indices": [], "error": str(e)}
    
    if prefix:
        indices = [idx for idx in indices if idx.startswith(prefix)]
    
    return {"indices": indices}


@router.delete("/elasticsearch/indices/batch")
def delete_elasticsearch_indices_by_prefix(prefix: str):
    config_path = 'data/base_config_server.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    params = config.get('params_vectorbase', {})
    es = Elasticsearch(
        [params.get('url', 'http://localhost:9200')],
        basic_auth=(params.get('auth', ['elastic', ''])[0], params.get('auth', ['', ''])[1]),
        verify_certs=False,
        ssl_show_warn=False
    )
    
    deleted = []
    try:
        for index_name in es.indices.get_alias(index='*').keys():
            if index_name.startswith(prefix):
                es.indices.delete(index=index_name)
                deleted.append(index_name)
    except Exception as e:
        return {"status": "error", "error": str(e), "deleted_count": len(deleted), "indices": deleted}
    
    return {"status": "deleted", "deleted_count": len(deleted), "indices": deleted}


@router.delete("/elasticsearch/indices/{index_name}")
def delete_elasticsearch_index(index_name: str):
    config_path = 'data/base_config_server.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    params = config.get('params_vectorbase', {})
    es = Elasticsearch(
        [params.get('url', 'http://localhost:9200')],
        basic_auth=(params.get('auth', ['elastic', ''])[0], params.get('auth', ['', ''])[1]),
        verify_certs=False,
        ssl_show_warn=False
    )
    
    try:
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
            return {"status": "deleted", "index": index_name}
    except Exception as e:
        return {"status": "error", "error": str(e), "index": index_name}
    return {"status": "not_found", "index": index_name}


@router.get("/custom")
def list_custom_rags():
    custom_rags_path = 'data/custom_rags'
    if not os.path.exists(custom_rags_path):
        return {"custom_rags": []}
    
    custom_rags = [f.replace('.json', '') for f in os.listdir(custom_rags_path) if f.endswith('.json')]
    return {"custom_rags": custom_rags}


@router.post("/custom")
def create_custom_rag(request: CustomRagRequest):
    custom_rags_path = 'data/custom_rags'
    os.makedirs(custom_rags_path, exist_ok=True)
    
    file_path = os.path.join(custom_rags_path, f"{request.name}.json")
    with open(file_path, 'w') as f:
        json.dump(request.config, f, indent=4, ensure_ascii=False)
    
    all_rags_path = 'data/all_rags.json'
    if os.path.exists(all_rags_path):
        with open(all_rags_path, 'r') as f:
            all_rags = json.load(f)
    else:
        all_rags = {}
    
    all_rags[request.name] = request.name
    with open(all_rags_path, 'w') as f:
        json.dump(all_rags, f, indent=4, ensure_ascii=False)
    
    return {"status": "created", "name": request.name}


@router.delete("/custom/{name}")
def delete_custom_rag(name: str):
    file_path = f'data/custom_rags/{name}.json'
    if os.path.exists(file_path):
        os.remove(file_path)
    
    all_rags_path = 'data/all_rags.json'
    if os.path.exists(all_rags_path):
        with open(all_rags_path, 'r') as f:
            all_rags = json.load(f)
        if name in all_rags:
            del all_rags[name]
        with open(all_rags_path, 'w') as f:
            json.dump(all_rags, f, indent=4, ensure_ascii=False)
    
    return {"status": "deleted", "name": name}


@router.get("/custom/{name}")
def get_custom_rag(name: str):
    file_path = f'data/custom_rags/{name}.json'
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Custom RAG not found")
    
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config


@router.get("/merge")
def list_merge_rags():
    merge_rags_path = 'data/merge'
    if not os.path.exists(merge_rags_path):
        return {"merge_rags": []}
    
    merge_rags = [f.replace('.json', '') for f in os.listdir(merge_rags_path) if f.endswith('.json')]
    return {"merge_rags": merge_rags}


@router.get("/merge/{name}")
def get_merge_rag(name: str):
    file_path = f'data/merge/{name}.json'
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Merge RAG not found")
    
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config


@router.post("/merge")
def create_merge_rag(request: MergeRagRequest):
    merge_rags_path = 'data/merge'
    os.makedirs(merge_rags_path, exist_ok=True)
    
    merge_config = request.config.copy()
    merge_config['name'] = request.name
    merge_config['base'] = 'merger'
    merge_config['rag_list'] = request.rag_list
    merge_config['rag_config_list'] = request.rag_config_list
    
    file_path = os.path.join(merge_rags_path, f"{request.name}.json")
    with open(file_path, 'w') as f:
        json.dump(merge_config, f, indent=4, ensure_ascii=False)
    
    all_rags_path = 'data/all_rags.json'
    if os.path.exists(all_rags_path):
        with open(all_rags_path, 'r') as f:
            all_rags = json.load(f)
    else:
        all_rags = {}
    
    all_rags[request.name] = request.name
    with open(all_rags_path, 'w') as f:
        json.dump(all_rags, f, indent=4, ensure_ascii=False)
    
    return {"status": "created", "name": request.name}


@router.delete("/merge/{name}")
def delete_merge_rag(name: str):
    file_path = f'data/merge/{name}.json'
    if os.path.exists(file_path):
        os.remove(file_path)
    
    all_rags_path = 'data/all_rags.json'
    if os.path.exists(all_rags_path):
        with open(all_rags_path, 'r') as f:
            all_rags = json.load(f)
        if name in all_rags:
            del all_rags[name]
        with open(all_rags_path, 'w') as f:
            json.dump(all_rags, f, indent=4, ensure_ascii=False)
    
    return {"status": "deleted", "name": name}
