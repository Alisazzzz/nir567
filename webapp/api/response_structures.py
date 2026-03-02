#All structures for server responses are here



#--------------------------
#---------imports----------
#--------------------------

from typing import Any, Dict, List, Optional
from pydantic import BaseModel



#---------------------------------------
#---------structures for graph----------
#---------------------------------------

class ExistingGraph(BaseModel):
    filename: str
    document: str
    is_current: bool

class SelectedGraph(BaseModel):
    filepath: str

class GraphInfo(BaseModel):
    document_filepath: str
    graph_filename: str
    embedding_model_name: str



#----------------------------------------
#---------structures for models----------
#----------------------------------------

class ExistingModel(BaseModel):
    name: str
    option: str
    model_name: str
    max_tokens: int
    temperature: float
    is_current: bool

class ExistingModelShort(BaseModel):
    name: str

class SelectedModel(BaseModel):
    model_name: str
    model_type: str

class ChatOrInstruct(BaseModel):
    model_type: str

class ModelToCreate(BaseModel):
    name: str
    option: str
    model_name: str
    max_tokens: int
    temperature: float
    api: Optional[str] = None
    model_type: str



#--------------------------------------------
#---------structures for embeddings----------
#--------------------------------------------

class ExistingEmbedding(BaseModel):
    name: str
    model_name: Optional[str]
    option: str

class SelectedEmbedding(BaseModel):
    name: str
    model_type: str

class EmbeddingsInfo(BaseModel):
    name: str
    option: str
    model_name: str
    api_info: Optional[str]




#--------------------------------------
#---------structures for chat----------
#--------------------------------------

class ChatMessage(BaseModel):
    text: str

class ChatResponse(BaseModel):
    answer: str
    model: str

class ChatHistory(BaseModel):
    history: List[Dict[str, str]]