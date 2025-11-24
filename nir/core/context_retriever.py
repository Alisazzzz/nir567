#All functions for context search are here

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from typing import List, Tuple, Optional

from nir.graph.knowledge_graph import KnowledgeGraph
from nir.graph.graph_structures import Node, Edge


def retrieve_similar_nodes(
        graph: KnowledgeGraph,
        text: str,
        embedding_model: Embeddings,
        threshold: float = 0.65
    ) -> List[Tuple[Node, float]]:
    
    query_emb = np.array(embedding_model.embed_query(text))
    candidates = []
    for node in graph.get_all_nodes():
        node_emb = np.array(embedding_model.embed_query(node.name))
        sim = float(np.dot(node_emb, query_emb) / (np.linalg.norm(node_emb) * np.linalg.norm(query_emb)))
        if sim >= threshold:
            candidates.append((node, sim))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def find_paths(
        graph: KnowledgeGraph,
        start_id: str,
        end_id: str,
        alpha=0.7,
        threshold=0.01,
        max_depth=5
    ) -> List[Tuple[List[Node], float]]:
    
    results = []
    queue = [(start_id, [start_id], 1.0)]
    while queue:
        current, path, S_current = queue.pop(0)
        print(current)
        if current == end_id:
            results.append((path, S_current))
            continue
        if len(path) > max_depth:
            continue
        outgoing_edges = [e for e in graph.get_all_edges() if e.source == current or e.target == current]
        print(outgoing_edges)
        out_degree = len(outgoing_edges)
        for edge in outgoing_edges:
            next_node = edge.target
            if any(n == next_node for n in path):
                continue
            S_next = alpha * (S_current / max(out_degree, 1))
            print(S_next)
            if S_next < threshold:
                continue
            new_path = path + [next_node]
            queue.append((next_node, new_path, S_next))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

def form_context_without_llm(
        query: str,
        graph: KnowledgeGraph,
        embedding_model: Embeddings,
        max_size: int
    ) -> str:

    result_nodes = [] # (Node, float)
    result_edges = [] # Edge
    result_paths = [] # (Node, Edge, )

    entry_nodes = retrieve_similar_nodes(graph, query, embedding_model)
    

    result = ""
    return result 

from nir.graph.graph_storages.networkx_graph import NetworkXGraph

graph_loaded = NetworkXGraph()
graph_loaded.load("assets/graphs/graph_map_short.json")
print(find_paths(graph_loaded, "mira_stone", "ariendale_village"))