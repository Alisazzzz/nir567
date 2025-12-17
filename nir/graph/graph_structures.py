#All data structures needed for graph are here



#--------------------------
#---------imports----------
#--------------------------

import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any


#--------------------------------------------------
#-----------information about node types-----------
#--------------------------------------------------

TYPE_CORRECTIONS = {
    "location_element": "location",
    "place": "location",
    "person": "character",
    "human": "character",
    "object": "item",
    "thing": "item",
    "env_element": "environment_element",
    "natural_feature": "environment_element",
}

POSSIBLE_TYPES = [
    "character",
    "group",
    "location",
    "environment_element",
    "item",
    "event"
]


#-----------------------------------------
#-----------structures for graph----------
#-----------------------------------------

class State(BaseModel):
    sid: str
    current_description: str = ""
    current_attributes: Dict[str, Any] = Field(default_factory=dict)
    time_start_event: Optional[str] = None
    time_end_event: Optional[str] = None

class Node(BaseModel):
    id: str
    name: str
    type: str
    base_description: str = ""
    base_attributes: Dict[str, Any] = Field(default_factory=dict) # time_start, time_end for events are required
    states: List[State] = Field(default_factory=list)
    chunk_id: List[int]

class Edge(BaseModel):
    id: str
    source: Optional[str] = None
    target: Optional[str] = None
    relation: str
    description: str = ""
    weight: float = 1.0
    time_start_event: Optional[str] = None
    time_end_event: Optional[str] = None
    chunk_id: int


#--------------------------------------------------------
#-----structures for first stage of graph extraction-----
#--------------------------------------------------------

class ExtractedNode(BaseModel):
    name: str
    type: str
    base_description: str = ""
    base_attributes: Dict[str, Any] = Field(default_factory=dict) # time_start, time_end for events are required

    @field_validator('type')
    @classmethod
    def validate_and_correct_type(cls, v: str) -> str:
        v_clean = v.strip().lower()        
        if v_clean in [item for item in POSSIBLE_TYPES]:
            return v_clean
        if v_clean in TYPE_CORRECTIONS:
            corrected = TYPE_CORRECTIONS[v_clean]
            return corrected
        return "item"

class ExtractedEdge(BaseModel):
    node1: Optional[str] = None
    node2: Optional[str] = None
    relation_from1to2: str
    relation_from2to1: str
    description: str = ""
    weight: float = 1.0

class GraphExtractionResult(BaseModel):
    nodes: List[ExtractedNode] = Field(default_factory=list)
    edges: List[ExtractedEdge] = Field(default_factory=list)


#-----------------------------------------------------------------
#-----structures for semi stage of graph extraction (merging)-----
#-----------------------------------------------------------------

class MergedNode(BaseModel):
    name: str
    base_description: str = ""
    base_attributes: Dict[str, Any] = Field(default_factory=dict) # time_start, time_end for events are required


#---------------------------------------------------------
#-----structures for second stage of graph extraction-----
#---------------------------------------------------------

class InputNode(BaseModel):
    id: str
    name: str
    base_description: str = ""
    base_attributes: Dict[str, Any] = Field(default_factory=dict) # time_start, time_end for events are required
    states: List[State] = Field(default_factory=list)

class InputEdge(BaseModel):
    id: str
    source: Optional[str] = None
    target: Optional[str] = None
    relation: str
    description: str = ""
    time_start_event: Optional[str] = None
    time_end_event: Optional[str] = None

class AffectedNode(BaseModel):
    id: str
    name: str
    new_current_description: str
    new_current_attributes: Dict[str, Any] = Field(default_factory=dict)
    time_start_event: Optional[str] = None
    time_end_event: Optional[str] = None

    @field_validator('new_current_description')
    @classmethod
    def ensure_string_field(cls, v):
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            try:
                return json.dumps(v, ensure_ascii=False, separators=(',', ':'))
            except:
                return str(v)
        if isinstance(v, (int, float, bool)):
            return str(v)
        return str(v)

class AffectedEdge(BaseModel):
    id: str
    new_description: str
    time_start_event: Optional[str] = None
    time_end_event: Optional[str] = None

    @field_validator('new_description')
    @classmethod
    def ensure_string_field(cls, v):
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            try:
                return json.dumps(v, ensure_ascii=False, separators=(',', ':'))
            except:
                return str(v)
        if isinstance(v, (int, float, bool)):
            return str(v)
        return str(v)

class EventImpact(BaseModel):
    event_name: str
    affected_nodes: List[AffectedNode] = Field(default_factory=list)
    affected_edges: List[AffectedEdge] = Field(default_factory=list)

class EventsSubgraph(BaseModel):
    events_with_impact: List[EventImpact] = Field(default_factory=list)

