#Abstract class for knowledge graph

from typing import List, Optional

from nir.graph.graph_structures import Node, Edge, State
from abc import ABC, abstractmethod

class KnowledgeGraph(ABC):
    @abstractmethod
    def add_node(self, node: Node) -> None:
        pass

    @abstractmethod
    def add_edge(self, edge: Edge) -> None:
        pass

    @abstractmethod
    def get_node_by_name(self, node_name: str) -> Node:
        pass

    @abstractmethod
    def get_node_by_id(self, node_id: str) -> Node:
        pass
    
    @abstractmethod
    def get_all_nodes(self) -> List[Node]:
        pass

    @abstractmethod
    def get_edges_between_nodes(self, source: str, target: str) -> List[Edge]: #between one pair of nodes
        pass

    @abstractmethod
    def get_edge_by_id(self, edge_id: str) -> Edge:
        pass

    @abstractmethod
    def get_all_edges(self) -> List[Edge]:
        pass

    @abstractmethod
    def update_node_state(self, node_id: str, new_desctiption: str, new_state: State) -> None:
        pass

    @abstractmethod
    def update_edge_times(self,  edge_id: str, time_start_event: Optional[str] = None, time_end_event: Optional[str] = None) -> None:
        pass
    
    @abstractmethod
    def update_node_full(self, node_id: str, new_info: Node) -> None:
        pass

    @abstractmethod
    def update_edge_full(self, edge_id: str, new_info: Edge) -> None:
        pass

    @abstractmethod
    def remove_node(self, node_id: str) -> None:
        pass

    @abstractmethod
    def remove_edge(self, edge_id: str) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def visualize(self) -> None:
        pass