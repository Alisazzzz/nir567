#All data structures needed for graph are here



#--------------------------
#---------imports----------
#--------------------------

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


#-----------------------------------------------------
#-----------additional extraction structures----------
#-----------------------------------------------------

class AffectedNode(BaseModel):
    id: str
    name: str
    new_current_description: str
    new_current_attributes: Dict[str, Any] = Field(default_factory=dict)
    time_start_event: Optional[str] = None
    time_end_event: Optional[str] = None

class AffectedEdge(BaseModel):
    id: str
    new_description: str
    time_start_event: Optional[str] = None
    time_end_event: Optional[str] = None

class EventImpact(BaseModel):
    event_name: str
    affected_nodes: List[AffectedNode] = Field(default_factory=list)
    affected_edges: List[AffectedEdge] = Field(default_factory=list)


class EventsSubgraph(BaseModel):
    events_with_impact: List[EventImpact] = Field(default_factory=list)

class GraphExtractionResult(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)