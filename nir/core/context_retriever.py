#All functions for context search are here

def get_context_networkX(query: str, retriever, G):
    search_results = retriever.invoke(query)
    entities = [res.page_content for res in search_results]

    results = {
        "query": query,
        "found_entities": entities,
        "entities_context": [],
        "relationships": [],
    }

    for entity in entities:
            
        entity_data = {
            "entity": entity,
            "node_attributes": dict(G.nodes[entity]),
            "neighbors": []
        }
        
        neighbors = list(G.neighbors(entity))
        
        for neighbor in neighbors:

            neighbor_info = {
                "neighbor": neighbor,
                "neighbor_attributes": dict(G.nodes[neighbor]),
                "edge_attributes": {}
            }
            
            if G.has_edge(entity, neighbor):
                edge_data = G[entity][neighbor]
                neighbor_info["edge_attributes"] = dict(edge_data)
                
                relationship = {
                    "source": entity,
                    "target": neighbor,
                    "attributes": dict(edge_data),
                    "type": edge_data.get('relationship_type', 'connected')
                }
                results["relationships"].append(relationship)
            entity_data["neighbors"].append(neighbor_info)
        results["entities_context"].append(entity_data)

    return results

def prepare_context(context: dict) -> str:
    formatted_lines = []
    for entity_ctx in context['entities_context']:
        entity = entity_ctx['entity']
        formatted_lines.append(f"\n{entity}:")
        
        if entity_ctx['node_attributes'] and entity_ctx['node_attributes'].get('properties') != '{}':
            props = entity_ctx['node_attributes']['properties']
            if props and props != '{}':
                formatted_lines.append(f"  Properties: {props}")
        
        for neighbor_info in entity_ctx['neighbors']:
            neighbor = neighbor_info['neighbor']
            edge_attrs = neighbor_info['edge_attributes']
            
            relation_type = edge_attrs.get('type', 'related_to')
            
            formatted_line = f"  {entity} - {relation_type} - {neighbor}"
            
            if edge_attrs.get('properties') and edge_attrs['properties'] != '{}':
                props = edge_attrs['properties']
                formatted_line += f" [{props}]"
            
            formatted_lines.append(formatted_line)
    
    return "\n".join(formatted_lines)


import spacy
from typing import List, Set, Optional
from nir.graph.graph_structures import Node
from nir.graph.knowledge_graph import KnowledgeGraph

_nlp = spacy.load("en_core_web_sm") 

def extract_entities_from_query(query: str) -> Set[str]:
    doc = _nlp(query)
    entities = {ent.text.strip().lower() for ent in doc.ents}
    return entities

def find_matching_nodes_in_graph(kg: KnowledgeGraph, query: str, case_sensitive: bool = False) -> List[Node]:
    entities = extract_entities_from_query(query)
    matched_nodes = []

    all_nodes = kg.get_all_nodes()

    for node in all_nodes:
        node_name = node.name if case_sensitive else node.name.lower()
        if node_name in entities:
            matched_nodes.append(node)

    return matched_nodes

#test

from nir.graph.graph_storages.networkx_graph import NetworkXGraph
graph_loaded = NetworkXGraph()
graph_loaded.load("assets/graphs/graph.json")

query = "Create a new character who is a friend of Ali Baba"
nodes = find_matching_nodes_in_graph(graph_loaded, query, False)
print(nodes)

#new variant

import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity

from nir.graph.graph_structures import Node, Edge, State, NodeType
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.embedding.vector_store import VectorStore
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
import json

def extract_entities(query: str, llm: BaseLanguageModel) -> List[str]:
    """Шаг 1: Извлечение сущностей из запроса с помощью LLM."""
    prompt = (
        "Extract named entities (characters, locations, items, events, groups, environment elements) "
        "from the following query. Return ONLY a JSON list of strings, nothing else.\n\n"
        f"Query: {query}"
    )
    try:
        raw = llm.invoke(prompt).content.strip()
        # Обработка случаев, когда LLM возвращает нечистый JSON
        if raw.startswith("```json"):
            raw = raw[7:-3].strip()
        entities = json.loads(raw)
        if isinstance(entities, list):
            return [str(e).strip() for e in entities if e]
    except Exception as e:
        print(f"[WARN] Entity extraction failed: {e}")
    return []

def generate_embedding(text: str, embedding_model: Embeddings) -> List[float]:
    """Генерация эмбеддинга текста."""
    return embedding_model.embed_query(text)

def filter_nodes_by_time(nodes: List[Node], time_window: Optional[Dict[str, str]]) -> List[Node]:
    """Фильтрация узлов по временному окну (на основе их состояний)."""
    if not time_window:
        return nodes

    start_event = time_window.get("start_event_id")
    end_event = time_window.get("end_event_id")

    def is_state_active(state: State) -> bool:
        # Состояние активно, если пересекается с [start_event, end_event]
        state_start = state.time_start
        state_end = state.time_end  # может быть None → бесконечно

        # Простейшая логика сравнения по ID событий как строкам (можно заменить на порядковые номера)
        # Здесь предполагается, что event_id — это строка, и есть способ их сравнивать (например, timestamp или int-as-str)
        # Для простоты будем считать, что сравнение строк корректно отражает хронологию
        # (в реальном проекте лучше использовать числовые метки времени)

        if end_event is None:
            window_end = float('inf')
        else:
            try:
                window_end = int(end_event)
            except:
                window_end = end_event

        try:
            s_start = int(state_start)
        except:
            s_start = state_start

        if state_end is None:
            s_end = float('inf')
        else:
            try:
                s_end = int(state_end)
            except:
                s_end = state_end

        try:
            w_start = int(start_event) if start_event else -float('inf')
        except:
            w_start = start_event if start_event else ""

        # Если нельзя сравнивать численно — пропускаем фильтрацию
        if isinstance(s_start, str) or isinstance(w_start, str):
            return True

        return not (s_end < w_start or s_start > window_end)

    filtered = []
    for node in nodes:
        if any(is_state_active(state) for state in node.states):
            filtered.append(node)
    return filtered

def filter_edges_by_time(edges: List[Edge], time_window: Optional[Dict[str, str]]) -> List[Edge]:
    """Фильтрация рёбер по временным меткам."""
    if not time_window:
        return edges

    start_event = time_window.get("start_event_id")
    end_event = time_window.get("end_event_id")

    def edge_overlaps(edge: Edge) -> bool:
        e_start = edge.time_start_event
        e_end = edge.time_end_event

        # Аналогично — упрощённая логика
        if e_start is None and e_end is None:
            return True
        if e_end is None and start_event is None:
            return True

        try:
            ew_start = int(e_start) if e_start else -float('inf')
            ew_end = int(e_end) if e_end else float('inf')
            w_start = int(start_event) if start_event else -float('inf')
            w_end = int(end_event) if end_event else float('inf')
        except:
            return True  # если не можем сравнить — оставляем

        return not (ew_end < w_start or ew_start > w_end)

    return [e for e in edges if edge_overlaps(e)]

def find_neighbors_of_nodes(graph: KnowledgeGraph, node_ids: Set[str]) -> List[Node]:
    """Возвращает список соседних узлов для заданных node_id."""
    neighbors = set()
    all_nodes = {node.id: node for node in graph.get_all_nodes()}
    for nid in node_ids:
        # Ищем все рёбра, где nid — источник или цель
        for edge in graph.get_all_edges():
            if edge.source == nid and edge.target in all_nodes:
                neighbors.add(edge.target)
            elif edge.target == nid and edge.source in all_nodes:
                neighbors.add(edge.source)
    return [all_nodes[nid] for nid in neighbors if nid in all_nodes]

def compute_path_nodes(graph: KnowledgeGraph, seed_nodes: List[Node]) -> Set[str]:
    """Находит узлы, лежащие на кратчайших путях между парами seed_nodes.
    Реализация через простой BFS между парами (можно заменить на networkx при необходимости)."""
    if len(seed_nodes) < 2:
        return set()

    # Построим временный networkx граф для поиска путей
    try:
        import networkx as nx
    except ImportError:
        print("[WARN] networkx not available — skipping path expansion")
        return set()

    G = nx.DiGraph()
    all_nodes = graph.get_all_nodes()
    node_id_set = {n.id for n in all_nodes}
    G.add_nodes_from(node_id_set)

    for edge in graph.get_all_edges():
        if edge.source in node_id_set and edge.target in node_id_set:
            G.add_edge(edge.source, edge.target, weight=edge.weight)

    path_nodes = set()
    ids = [n.id for n in seed_nodes]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            try:
                path = nx.shortest_path(G, source=ids[i], target=ids[j])
                path_nodes.update(path)
            except nx.NetworkXNoPath:
                continue
    return path_nodes

def estimate_size(node_or_edge: Any) -> int:
    """Оценка размера в символах (можно заменить на токены)."""
    if isinstance(node_or_edge, Node):
        return len(node_or_edge.name) + len(node_or_edge.description)
    elif isinstance(node_or_edge, Edge):
        return len(node_or_edge.relation) + len(node_or_edge.description)
    return 0

def retrieve_context(
    graph: KnowledgeGraph,
    vector_store: VectorStore,
    embedding_model: Embeddings,
    llm: BaseLanguageModel,
    query: str,
    theta: float = 0.6,
    context_token_limit: int = 3000,
    time_window: Optional[Dict[str, str]] = None  # {"start_event_id": "...", "end_event_id": "..."}
) -> List[Dict[str, Any]]:
    """
    Реализует алгоритм из раздела 1.4 документа.
    Возвращает список элементов контекста: {'element': Node/Edge, 'relevance': float, 'size': int}
    """
    # Шаг 1: Извлечение сущностей
    entities = extract_entities(query, llm)
    query_emb = generate_embedding(query, embedding_model)

    # Шаг 2: Эмбеддинги сущностей
    search_texts = entities if entities else [query]
    search_embs = [generate_embedding(text, embedding_model) for text in search_texts]

    # Шаг 3a: Поиск близких сущностей через векторное хранилище
    # Предполагаем, что vector_store поддерживает similarity_search_by_vector
    # Если нет — можно добавить метод в абстракцию
    from langchain_core.vectorstores import VectorStore as LangChainVectorStore
    if hasattr(vector_store, "similarity_search_by_vector"):
        # Используем LangChain-совместимый метод
        J_nodes = set()
        for emb in search_embs:
            results = vector_store.similarity_search_by_vector(emb, k=10, score_threshold=1 - theta)
            for doc in results:
                meta = doc.metadata
                if meta.get("type") == "node":
                    node = graph.get_node_by_id(meta["id"])
                    J_nodes.add(node)
    else:
        # Fallback: перебор всех узлов (медленно!)
        print("[WARN] VectorStore lacks similarity_search_by_vector — using brute-force")
        all_nodes = graph.get_all_nodes()
        J_nodes = set()
        for node in all_nodes:
            node_text = f"{node.name}. {node.description}"
            node_emb = generate_embedding(node_text, embedding_model)
            for emb in search_embs:
                sim = cosine_similarity([emb], [node_emb])[0][0]
                if sim >= theta:
                    J_nodes.add(node)
                    break

    J_nodes = list(J_nodes)

    # Шаг 3d: Фильтрация по времени (узлы)
    J_nodes = filter_nodes_by_time(J_nodes, time_window)

    # Шаг 3b: Окрестности N
    J_ids = {n.id for n in J_nodes}
    N_nodes = find_neighbors_of_nodes(graph, J_ids)
    N_nodes = filter_nodes_by_time(N_nodes, time_window)

    # Шаг 3c: Пути P
    P_node_ids = compute_path_nodes(graph, J_nodes)
    all_nodes_dict = {n.id: n for n in graph.get_all_nodes()}
    P_nodes = [all_nodes_dict[nid] for nid in P_node_ids if nid in all_nodes_dict]
    P_nodes = filter_nodes_by_time(P_nodes, time_window)

    # Собираем все кандидаты
    candidate_nodes = list(set(J_nodes + N_nodes + P_nodes))

    # Подготавливаем эмбеддинги узлов для релевантности
    node_to_emb = {}
    for node in candidate_nodes:
        text = f"{node.name}. {node.description}"
        node_to_emb[node.id] = generate_embedding(text, embedding_model)

    # Вычисляем релевантность по формуле (17)
    scored_items = []
    for node in candidate_nodes:
        emb = node_to_emb[node.id]
        cos_sim = cosine_similarity([emb], [query_emb])[0][0]

        is_neighbor = node in N_nodes
        # Найдём максимальный вес связи с J
        max_weight = 0.0
        for j_node in J_nodes:
            edges = graph.get_edges_between_nodes(j_node.id, node.id)
            if not edges:
                edges = graph.get_edges_between_nodes(node.id, j_node.id)
            if edges:
                max_weight = max(max_weight, max(e.weight for e in edges))

        is_on_path = node in P_nodes
        path_score = 0.5  # можно параметризовать

        relevance = cos_sim + (max_weight if is_neighbor else 0.0) + (path_score if is_on_path else 0.0)

        scored_items.append({
            "element": node,
            "relevance": relevance,
            "size": estimate_size(node)
        })

    # Сортировка по релевантности (шаг 6)
    scored_items.sort(key=lambda x: x["relevance"], reverse=True)

    # Ограничение по размеру (шаг 7)
    context = []
    total_size = 0
    for item in scored_items:
        if total_size + item["size"] <= context_token_limit:
            context.append(item)
            total_size += item["size"]
        else:
            break

    return context