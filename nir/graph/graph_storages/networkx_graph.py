#In case of using NetworkX as graph storage

import networkx as nx
from typing import List
from langchain_community.graphs.graph_document import GraphDocument

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