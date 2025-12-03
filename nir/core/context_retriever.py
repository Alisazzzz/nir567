#All functions for context search are here

import numpy as np
import re
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from collections import defaultdict, deque

from nir.graph.knowledge_graph import KnowledgeGraph
from nir.graph.graph_structures import Node, Edge, NodeType


NODE_RATIO = 0.5
EDGE_RATIO = 0.3
PATH_RATIO = 0.2


def extract_event_sequence(graph) -> Dict[str, int]:
    events = {n.id: n for n in graph.get_all_nodes() if n.type == NodeType.event}
    edges_and_followers = defaultdict(list)
    amount_of_precedors = {eid: 0 for eid in events}

    for edge in graph.get_all_edges():
        if edge.source in events and edge.target in events:
            if edge.relation == "precedes":
                edges_and_followers[edge.source].append(edge.target)
                amount_of_precedors[edge.target] += 1
            elif edge.relation == "follows":
                edges_and_followers[edge.target].append(edge.source)
                amount_of_precedors[edge.source] += 1
    queue = deque([eid for eid in events if amount_of_precedors[eid] == 0])
    stage = 0
    event_stage = {}

    while queue:
        next_queue = deque()
        for eid in queue:
            event_stage[eid] = stage
            for nxt in edges_and_followers[eid]:
                amount_of_precedors[nxt] -= 1
                if amount_of_precedors[nxt] == 0:
                    next_queue.append(nxt)
        queue = next_queue
        stage += 1
    if len(event_stage) != len(events):
        return {}
    return event_stage


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


def filter_graph_by_events(
        nodes: List[Node],
        edges: List[Edge],
        event_sequence: Dict[str, int],
        start_event_id: Optional[str] = None,
        end_event_id: Optional[str] = None
    ) -> Tuple[List[Node], List[Edge]]:

    min_time = 0
    max_time = max(event_sequence.values()) if event_sequence else 0
    left = event_sequence.get(start_event_id, min_time) if start_event_id else min_time
    right = event_sequence.get(end_event_id, max_time) if end_event_id else max_time
    if left > right:
        left, right = right, left

    def overlaps(start_eid, end_eid):
        start_level = event_sequence.get(start_eid, min_time) if start_eid else min_time
        end_level = event_sequence.get(end_eid, max_time) if end_eid else max_time
        return not (end_level < left or start_level > right)

    filtered_nodes = []
    for node in nodes:
        new_states = []
        for st in node.states:
            if overlaps(st.time_start, st.time_end):
                new_states.append(st)
        new_node = node.model_copy(deep=True)
        new_node.states = new_states
        filtered_nodes.append(new_node)

    filtered_edges = []
    for edge in edges:
        if overlaps(edge.time_start_event, edge.time_end_event):
            filtered_edges.append(edge)
    return filtered_nodes, filtered_edges


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
                edge = graph.get_edge_by_id(edge_id)
                pretty_parts.append(f" - {edge.relation} - ")
        pretty_string = "".join(pretty_parts)
        result_paths_dict[(source_id, target_id, index)] = (pretty_string, score)

    result_nodes.sort(key=lambda x: x[1])
    for node, weight in result_nodes:
        text = f"{node.name}. {node.description}"
        tokens = estimate_tokens(text)
        result_nodes_dict[node.id] = (text, tokens)
    
    node_budget = int(max_tokens * NODE_RATIO)
    edge_budget = int(max_tokens * EDGE_RATIO)
    path_budget = int(max_tokens * PATH_RATIO)

    selected_nodes = set()
    used_tokens = 0
    for node_id, (text, tokens) in result_nodes_dict.items():
        if used_tokens + tokens <= node_budget:
            selected_nodes.add(node_id)
            used_tokens += tokens
    
    filtered_edges = [(pair, (text, tokens)) for pair, (text, tokens) in result_edges_dict.items() if pair[0] in selected_nodes and pair[1] in selected_nodes]
    selected_edges = []
    edge_tokens = 0
    for pair, (text, tokens) in filtered_edges:
        if edge_tokens + tokens <= edge_budget:
            selected_edges.append((pair, text))
            edge_tokens += tokens

    filtered_paths = [(key, result_paths_dict[key]) for key in result_paths_dict.keys() if key[0] in selected_nodes and key[1] in selected_nodes]
    selected_paths = []
    path_tokens = 0
    for key, (text, tokens) in filtered_paths:
        if path_tokens + tokens <= path_budget:
            selected_paths.append((key, text))
            path_tokens += tokens
    
    result = "NODES:\n\n"
    for node_id in selected_nodes:
        text = result_nodes_dict[node_id][0]
        result += text + "\n"
    result += "\nEDGES:\n\n"
    for (_, text) in selected_edges:
        result += text + "\n"
    result += "\nPATHS:\n\n"
    for (_, text) in selected_paths:
        result += text + "\n"

    return result 

from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.llm.manager import ModelManager

graph_loaded = NetworkXGraph()
graph_loaded.load("assets/graphs/graph_map_short.json")

manager = ModelManager()
embedding_model = manager.create_embedding_model(name="embeddings", option="ollama", model_name="nomic-embed-text:v1.5")

print(form_context_without_llm("Who is Elias Thorn and what he do in Ariendale village?", graph_loaded, embedding_model))