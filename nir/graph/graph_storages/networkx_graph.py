#In case of using NetworkX as graph storage

import networkx as nx
from typing import List
from langchain_community.graphs.graph_document import GraphDocument
import json

from nir.graph.knowledge_graph import KnowledgeGraph
from nir.graph.graph_structures import Node, Edge, State

def create_networkX_graph(graph: List[GraphDocument]):
    
    G = nx.DiGraph()

    for doc in graph:
        for node in doc.nodes:
            G.add_node(node.id, type=node.type, properties=node.properties)
        for relationship in doc.relationships:
            G.add_edge(relationship.source.id, relationship.target.id, type=relationship.type, properties=relationship.properties)

    for node in G.nodes():
        node_attrs = G.nodes[node]
        for attr, value in list(node_attrs.items()):
            if isinstance(value, dict):
                import json
                G.nodes[node][attr] = json.dumps(value)

    for u, v in G.edges():
        edge_attrs = G.edges[u, v]
        for attr, value in list(edge_attrs.items()):
            if isinstance(value, dict):
                import json
                G.edges[u, v][attr] = json.dumps(value)

    nx.write_graphml_lxml(G, "assets/graphs/graph.graphml")
    
    return

class NetworkXKnowledgeGraph(KnowledgeGraph):
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

    def get_node(self, node_id: str) -> Node:
        if node_id not in self.graph.nodes:
            raise KeyError(f"Node {node_id} not found")
        data = self.graph.nodes[node_id]["data"]
        return Node(**data)

    def get_edges(self, source: str, target: str) -> List[Edge]:
        if not self.graph.has_edge(source, target):
            return []
        edges = []
        for key, attrs in self.graph[source][target].items():
            edge_data = attrs["data"]
            edges.append(Edge(**edge_data))
        return edges

    def update_node_state(self, node_id: str, new_state: State) -> None:
        node = self.get_node(node_id)
        node.states.append(new_state)
        self.graph.nodes[node_id]["data"] = node.model_dump()

    def save(self, filepath: str) -> None:
        data = {
            "nodes": [self.graph.nodes[n]["data"] for n in self.graph.nodes],
            "edges": [
                attrs["data"]
                for u, v, attrs in self.graph.edges(data=True)
            ]
        }
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