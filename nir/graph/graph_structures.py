#All data structures needed for graph are here

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum

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

class NodeType(str, Enum):
    character = "character"
    group = "group"
    location = "location"
    environment_element = "environment_element"
    item = "item"
    event = "event"

class State(BaseModel):
    sid: str
    attributes: Dict[str, Any]
    time_start: Optional[str] = None
    time_end: Optional[str] = None

class Node(BaseModel):
    id: str
    name: str
    type: NodeType
    synonyms: List[str] = Field(default_factory=list)
    description: str = ""
    attributes: Dict[str, Any] = Field(default_factory=dict)
    states: List[State] = Field(default_factory=list)
    chunk_id: List[int]

    @field_validator('type')
    @classmethod
    def validate_and_correct_type(cls, v: str) -> str:
        v_clean = v.strip().lower()        
        if v_clean in [item.value for item in NodeType]:
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


class AffectedNode(BaseModel):
    id: str
    name: str
    description: str
    attributes: Dict[str, Any] = Field(default_factory=dict)

class AffectedEdge(BaseModel):
    id: str
    description: str
    time_start_event: Optional[str] = None
    time_end_event: Optional[str] = None

class EventImpact(BaseModel):
    event_id: str
    description: str
    affected_nodes: Optional[List[AffectedNode]]
    affected_edges: Optional[List[AffectedEdge]]
    time_start: Optional[str] = None
    time_end: Optional[str] = None

class EventsSubgraph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    events_with_impact: List[EventImpact] = Field(default_factory=list)