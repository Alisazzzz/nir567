#All structures for server responses are here



#--------------------------
#---------imports----------
#--------------------------

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field



#---------------------------------------
#---------structures for graph----------
#---------------------------------------

class ExistingGraph(BaseModel):
    filename: str
    document: Optional[str] = "doc"
    is_current: bool

class SelectedGraph(BaseModel):
    filepath: str

class GraphInfo(BaseModel):
    document_filepath: str
    graph_filename: str
    embedding_model_name: str



#-----------------------------------------------------
#---------structures for graph visualization----------
#-----------------------------------------------------

class VisualizationNodeInfo(BaseModel):
    id: str
    name: str
    type: str

class VisualizationEdgeInfo(BaseModel):
    id: str
    relation: str
    source: Optional[str]
    target: Optional[str]
    weight: float = 1.0

class GraphVisualizationInfo(BaseModel):
    nodes: List[VisualizationNodeInfo]
    edges: List[VisualizationEdgeInfo]

class State(BaseModel):
    current_description: str = ""
    current_attributes: Dict[str, Any]
    time_start_event: Optional[str] = None
    time_end_event: Optional[str] = None

class DetailedNodeInfo(BaseModel): 
    id: str
    name: str
    type: str
    base_description: str = ""
    base_attributes: Optional[Dict[str, Any]] = None
    states: List[State] = Field(default_factory=list)

class DetailedEdgeInfo(BaseModel):
    id: str
    source: Optional[str]
    target: Optional[str]
    relation: str
    description: str = ""
    weight: float = 1.0
    time_start_event: Optional[str] = None
    time_end_event: Optional[str] = None

class ChosenElement(BaseModel):
    id: str


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

class SelectedModel(BaseModel):
    name: str
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
    use_timestamps: Optional[bool] = False
    add_history: Optional[bool] = False


class ChatResponse(BaseModel):
    answer: str
    model: str

class ChatHistory(BaseModel):
    history: List[Dict[str, str]]