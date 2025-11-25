#All functions for context search are here

import numpy as np
import re
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from typing import List, Tuple, Optional
from itertools import combinations

from nir.graph.knowledge_graph import KnowledgeGraph
from nir.graph.graph_structures import Node, Edge

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chunks = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    token_count = 0
    for chunk in chunks:
        length = len(chunk)
        if re.match(r"[A-Za-z0-9]+$", chunk):
            token_count += max(1, round(length / 3.5))
        elif re.match(r"[А-Яа-яЁё]+$", chunk):
            token_count += max(1, round(length / 2.4))
        else:
            token_count += 1
    return token_count


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
    ) -> List[Tuple[List[str], List[str], float]]:
    
    results = []
    queue = [(start_id, [start_id], [], 1.0)]
    while queue:
        current, path, edges, S_current = queue.pop(0)
        if current == end_id:
            results.append((path, edges, S_current))
            continue
        if len(path) > max_depth:
            continue
        outgoing_edges = [e for e in graph.get_all_edges() if e.source == current or e.target == current]
        out_degree = len(outgoing_edges)
        for edge in outgoing_edges:
            if edge.target == current:
                next_node = edge.source
            else:
                next_node = edge.target
            if any(n == next_node for n in path):
                continue
            S_next = alpha * (S_current / max(out_degree, 1))
            if S_next < threshold:
                continue
            new_path = path + [next_node]
            new_edges = edges + [edge.id]
            queue.append((next_node, new_path, new_edges, S_next))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def form_context_without_llm(
        query: str,
        graph: KnowledgeGraph,
        embedding_model: Embeddings,
        max_tokens: int = 1024
    ) -> str:

    result_nodes_dict = {} # { node_id : ("node_name. node_description", token_amount) }
    result_edges_dict = {} # { (source_id, target_id) : ("name_source - relation - name_target", token_amount) }
    result_paths_dict = {} # { (source_id, target_id, number) : ("name_node_start - relation - ... - relation - name_node_target", token_amount) }

    entry_nodes = retrieve_similar_nodes(graph, query, embedding_model)
    result_nodes = entry_nodes.copy() 
    for entry_node, resource in entry_nodes:
        neighbours = graph.get_neighbours_of_node(entry_node.id)
        for neighbour in neighbours:
            if not any(node[0] == neighbour for node in result_nodes):
                edges = [edge for edge in graph.get_all_edges() if edge.source == neighbour.id or edge.target == neighbour.id]
                edges = [edge for edge in edges if edge.source == entry_node.id or edge.target == entry_node.id]
                edges.sort(key=lambda x: x.weight)
                weight = edges[0].weight
                result_nodes.append((neighbour, weight))
                for edge in edges:
                    text = graph.get_node_by_id(edge.source).name + " - " + graph.get_node_by_id(edge.target).name + ". " + edge.description
                    tokens = estimate_tokens(text)
                    result_edges_dict[(edge.source, edge.target)] = (text, tokens)

    paths_nodes = []
    if len(entry_nodes) > 1:
        for a, b in combinations(entry_nodes, 2):
            paths_nodes.append(find_paths(graph, a[0].id, b[0].id))
    all_paths = [p for paths in paths_nodes for p in paths]
    for path in all_paths:
        i = 0
        for node_id in path[0]:
            node_in_path = graph.get_node_by_id(node_id)
            if not any(node[0] == node_in_path for node in result_nodes):
                result_nodes.append((node_in_path, path[2]))
    
    pair_counters = {}
    for path_nodes, path_edges, score in all_paths:
        source_id = path_nodes[0]
        target_id = path_nodes[-1]
        pair_key = (source_id, target_id)
        index = pair_counters.get(pair_key, 0)
        pair_counters[pair_key] = index + 1

        pretty_parts = []
        for i, node_id in enumerate(path_nodes):
            node = graph.get_node_by_id(node_id)
            pretty_parts.append(node.name)
            if i < len(path_edges):
                edge_id = path_edges[i]
                print(edge_id)
                edge = graph.get_edge_by_id(edge_id)
                pretty_parts.append(f" - {edge.relation} - ")
        pretty_string = "".join(pretty_parts)
        result_paths_dict[(source_id, target_id, index)] = (pretty_string, score)
    print(result_paths_dict)
    result_nodes.sort(key=lambda x: x[1])
    
    result = ""
    nodes = [node for node, resource in result_nodes]
    for node in nodes:
        result += f"{node.name}. {node.description}\n"
    return result 

from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.llm.manager import ModelManager

graph_loaded = NetworkXGraph()
graph_loaded.load("assets/graphs/graph_map_short.json")

manager = ModelManager()
embedding_model = manager.create_embedding_model(name="embeddings", option="ollama", model_name="nomic-embed-text:v1.5")

print(form_context_without_llm("Who is Elias Thorn and what he do in Ariendale village?", graph_loaded, embedding_model))