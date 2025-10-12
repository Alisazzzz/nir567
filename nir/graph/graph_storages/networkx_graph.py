#In case of using NetworkX as graph storage

import networkx as nx
from typing import List, Optional
from langchain_community.graphs.graph_document import GraphDocument
import json
from pyvis.network import Network

from nir.graph.knowledge_graph import KnowledgeGraph
from nir.graph.graph_structures import Node, Edge, State


class NetworkXGraph(KnowledgeGraph):
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_node(self, node: Node) -> None:
        self.graph.add_node(
            node.id,
            data=node.model_dump()
        )

    def add_edge(self, edge: Edge) -> None:
        self.graph.add_edge(
            edge.source,
            edge.target,
            key=edge.relation,
            data=edge.model_dump()
        )

    def get_node_by_name(self, node_name: str) -> Node:
        for node_id, node_attrs in self.graph.nodes(data=True):
            data = node_attrs.get("data")
        if data and data.get("name") == node_name:
            return Node(**data)
        return
    
    def get_node_by_id(self, node_id: str) -> Node:
        if node_id not in self.graph.nodes:
            return
        data = self.graph.nodes[node_id]["data"]
        return Node(**data)

    def get_all_nodes(self) -> List[Node]:
        nodes = []
        for node_id in self.graph.nodes:
            attrs = self.graph.nodes[node_id]
            if "data" in attrs:
                nodes.append(Node(**attrs["data"]))
            else:
                continue
        return nodes

    def get_edges_between_nodes(self, source: str, target: str) -> List[Edge]:
        if not self.graph.has_edge(source, target):
            return []
        edges = []
        for key, attrs in self.graph[source][target].items():
            edge_data = attrs["data"]
            edges.append(Edge(**edge_data))
        return edges

    def get_edge_by_id(self, edge_id: str) -> Edge:
        data = self.graph.edges[edge_id]["data"]
        return Edge(**data)

    def get_all_edges(self) -> List[Edge]:
        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            if "data" in attrs:
                edges.append(Edge(**attrs["data"]))
        return edges
    
    def update_node_state(self, node_id: str, new_desctiption: str, new_state: State) -> None:
        node = self.get_node(node_id)
        node.description = new_desctiption
        node.states.append(new_state)
        self.graph.nodes[node_id]["data"] = node.model_dump()

    def update_edge_times(self, edge_id: str, time_start_event: Optional[str] = None, time_end_event: Optional[str] = None) -> None:
        edge = self.get_edge_by_id(edge_id)
        if time_start_event:
            edge["data"]["time_start_event"] = time_start_event
        if time_end_event:
            edge["data"]["time_end_event"] = time_end_event
        self.graph.edges[edge_id]["data"] = edge
        return

    def remove_node(self, node_id: str) -> None:
        return

    def remove_edge(self, edge_id: str) -> None:
        return

    def save(self, filepath: str) -> None:
        nodes_data = []
        for n in self.graph.nodes:
            node_attrs = self.graph.nodes[n]
            if "data" not in node_attrs:
                node_attrs["data"] = {
                    "id": n,
                    "name": n,
                    "type": "item",
                    "description": "Auto-created missing node",
                    "attributes": {},
                    "states": []
                }
            nodes_data.append(node_attrs["data"])
        edges_data = []
        for u, v, attrs in self.graph.edges(data=True):
            if "data" not in attrs:            
                continue
            edges_data.append(attrs["data"])
        data = {"nodes": nodes_data, "edges": edges_data}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str) -> None:
        self.graph.clear()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for node_dict in data["nodes"]:
            node = Node(**node_dict)
            self.add_node(node)
        for edge_dict in data["edges"]:
            edge = Edge(**edge_dict)
            self.add_edge(edge)

    def visualize(self) -> None:
        graph_vis = self.graph.copy()
        for u, v, data in graph_vis.edges(data=True):
            edge_label = data['data']['relation']
            data['label'] = edge_label
        net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
        net.from_nx(graph_vis)
        net.set_options("""
            var options = {
            "nodes": {
                "font": {
                "size": 14
                }
            },
            "edges": {
                "color": {
                "inherit": true
                },
                "smooth": false,
                "font": {
                "size": 12,
                "color": "#ffffff"
                }
            },
            "physics": {
                "enabled": true,
                "stabilization": {
                "iterations": 100
                }
            }
            }
        """)
        net.show("assets/outputs/graph.html", notebook=False)
        return
