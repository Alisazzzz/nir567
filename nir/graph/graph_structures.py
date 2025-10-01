#All data structures needed for graph are here

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


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
    time_start: str
    time_end: Optional[str] = None

class Node(BaseModel):
    id: str
    name: str
    type: NodeType
    description: str = ""
    attributes: Dict[str, Any] = Field(default_factory=dict)
    states: List[State] = Field(default_factory=list)

class Edge(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    description: str = ""
    weight: float = 1.0
    time_start_event: Optional[str] = None
    time_end_event: Optional[str] = None