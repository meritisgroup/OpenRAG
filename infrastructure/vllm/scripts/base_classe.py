# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:18:36 2025

@author: chardy
"""
from fastapi import UploadFile, File
from pydantic import BaseModel
from typing import Optional, List


class LoadModelBase(BaseModel):
    model_name: str
    gpu_memory_utilization: Optional[float] = 0.45
    
class PredictBase(BaseModel):
    systems: List[str]
    prompts: List[str]
    model_name: str
    temperature: Optional[float] = 0

class ReleaseBase(BaseModel):
    model_name: Optional[str] = None


class EmbeddingsBase(BaseModel):
    texts: List[str]
    model_name: str


class RerankingBase(BaseModel):
    contexts: List[str]
    query: str
    model_name: str

class HFKey(BaseModel):
    key : str
