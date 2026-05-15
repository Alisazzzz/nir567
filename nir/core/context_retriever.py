#All functions for context search are here



#--------------------------
#---------imports----------
#--------------------------

import heapq
import json
import random
import spacy
import numpy as np
import re
from rapidfuzz import process
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from typing import Any, Dict, List, Tuple, Optional
from itertools import combinations
from collections import defaultdict, deque
from pydantic import BaseModel

from nir.graph.knowledge_graph import KnowledgeGraph
from nir.graph.graph_structures import Node, Edge
from nir.prompts import retrieval_prompts



#--------------------------
#-----additional stuff-----
#--------------------------

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

def remove_comments(s: str) -> str:
    out_chars = []
    i = 0
    n = len(s)
    in_string = False
    string_quote = ""
    in_single_line_comment = False
    in_multi_line_comment = False
    while i < n:
        c = s[i]

        if in_single_line_comment:
            if c == "\n":
                in_single_line_comment = False
                out_chars.append(c)
            i += 1
            continue
        
        if in_multi_line_comment:
            if c == "*" and i + 1 < n and s[i + 1] == "/":
                in_multi_line_comment = False
                i += 2
            else:
                i += 1
            continue
        
        if in_string:
            if c == "\\":
                if i + 1 < n:
                    out_chars.append(c)
                    out_chars.append(s[i + 1])
                    i += 2
                else:
                    out_chars.append(c)
                    i += 1
                continue
            elif c == string_quote:
                out_chars.append(c)
                in_string = False
                string_quote = ""
                i += 1
                continue
            else:
                out_chars.append(c)
                i += 1
                continue

        if c == '"' or c == "'":
            in_string = True
            string_quote = c
            out_chars.append(c)
            i += 1
            continue

        if c == "/" and i + 1 < n and s[i + 1] == "/":
            in_single_line_comment = True
            i += 2
            continue

        if c == "/" and i + 1 < n and s[i + 1] == "*":
            in_multi_line_comment = True
            i += 2
            continue

        if c == "#":
            prev = s[i - 1] if i - 1 >= 0 else "\n"
            if prev in {"\n", "\r", "\t", " ", ""}:
                in_single_line_comment = True
                i += 1
                continue
            else:
                out_chars.append(c)
                i += 1
                continue

        out_chars.append(c)
        i += 1
    return "".join(out_chars)

def extract_last_json(text: str) -> str:
    stack = 0
    start = None
    last = None
    for i, ch in enumerate(text):
        if ch == '{':
            if stack == 0:
                start = i
            stack += 1
        elif ch == '}':
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    last = text[start:i+1]
    return last

def clean_json(text: str) -> str:

    fallback = {
        "extracted_entities": [],
        "downer_border_event_name": None,
        "upper_border_event_name": None
    }
    
    def try_parse(candidate: str) -> str | None:
        try:
            cleaned = remove_comments(candidate)
            json.loads(cleaned)
            return cleaned
        except (json.JSONDecodeError, Exception):
            return None
    
    codeblock_match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if codeblock_match:
        possible_json = codeblock_match.group(1).strip()
        cleaned = remove_comments(possible_json)
        return cleaned

    balanced = extract_last_json(text)
    if balanced:
        try:
            cleaned = remove_comments(balanced)
            json.loads(balanced)
            return cleaned
        except json.JSONDecodeError:
            pass

    cleaned = re.sub(r"^[^{]+", "", text)
    cleaned = re.sub(r"[^}]+$", "", cleaned)
    cleaned = remove_comments(cleaned)
    cleaned = re.sub(r'(":\s*"[^"]*")\s*\([^)]*\)', r'\1', cleaned)
    result = try_parse(cleaned)
    if result:
        return result
    else:
        return json.dumps(fallback, ensure_ascii=False)

def clean_string(text: str) -> str:
    return re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text)

entities_from_graph = set()
def update_entities_from_graph(graph: KnowledgeGraph) -> None:
    global entities_from_graph
    entities_from_graph.clear()
    entities_from_graph = set([ node.name for node in graph.get_all_nodes() ])

nlp_models = {}
STOP_WORDS_EN = {"who", "what", "where", "when", "why", "how", "someone", "something", "anything", "everything", "he", "she", "they", "it", "them", "his", "her", "their", "him"}
STOP_WORDS_RU = {"кто", "что", "где", "когда", "почему", "зачем", "как", "откуда", "куда", "кто-то", "кто-нибудь", "что-то", "что-нибудь", "кое-что", "нечто", "всё", "все", "вся", "всё", "всего", "всему", "всем", "всём", "всякий", "любой", "каждый", "он", "она", "оно", "они", "мы", "ты", "вы", "я", "его", "её", "их", "мой", "твой", "наш", "ваш", "свой", "этот", "тот", "такой", "та", "те", "эти", "тех", "этом"}

def extract_mentioned_entities(query: str, already_extracted_entities: List[str] = [], similarity_threshold: float = 0.95, language: str = "en") -> List[str]:
    global entities_from_graph
    if language not in nlp_models:
        nlp_model = "ru_core_news_sm" if language == "ru" else "en_core_web_sm"
        nlp_models[language] = spacy.load(nlp_model)
    nlp = nlp_models[language]
    doc = nlp(query)

    if language == "ru":
        current_stop_words = STOP_WORDS_RU
    else:
        current_stop_words = STOP_WORDS_EN

    entities = set([word.lower() for word in already_extracted_entities])
    words = set([word.lower().strip() for word in query.split()])
    lowered_entities_from_graph = set([entity.lower() for entity in entities_from_graph])
    entities |= lowered_entities_from_graph & words
    for token in doc:
        if token.pos_ == "PROPN":
            entities.add((token.text).lower())
    for chunk in doc.noun_chunks:
        if not clean_string(chunk.text).lower().strip() in current_stop_words:
            entities.add(clean_string(chunk.text).lower().strip())
            
    final_extracted_entities_names = set()
    for extracted_entity in entities:
        match = process.extractOne(extracted_entity, entities_from_graph)
        if (match):
            name, score, index = match
            if (score > similarity_threshold):
                final_extracted_entities_names.add(name)

    print("FINAL ENTITIES --> ", final_extracted_entities_names)
    return list(final_extracted_entities_names)

def filter_node_states_by_time(
        node: Node,
        event_sequence: Dict[str, int], #event_id: number in sequence
        downer_border_event_id: str = None,
        upper_border_event_id: str = None
    ) -> Node:

    min_time = 0
    max_time = max(event_sequence.values())+1 if event_sequence else 0
    left = event_sequence.get(downer_border_event_id, min_time) if downer_border_event_id else min_time
    right = event_sequence.get(upper_border_event_id, max_time) if upper_border_event_id else max_time
    if left > right:
        left, right = right, left

    def overlaps(start_eid, end_eid):
        start_level = event_sequence.get(start_eid, min_time) if start_eid else min_time
        end_level = event_sequence.get(end_eid, max_time) if end_eid else max_time
        return not (start_level >= right or end_level < left)

    new_states = []
    for st in node.states:
        if overlaps(st.time_start_event, st.time_end_event):
            new_states.append(st)
    new_node = node.model_copy(deep=True)
    new_node.states = new_states
    return new_node



#-------------------------
#-----ratio variables-----
#-------------------------

NODE_RATIO_WITHOUT_HISTORY = 0.5
EDGE_RATIO_WITHOUT_HISTORY = 0.4
PATH_RATIO_WITHOUT_HISTORY = 0.1

NODE_RATIO_WITH_HISTORY = 0.4
EDGE_RATIO_WITH_HISTORY = 0.3
PATH_RATIO_WITH_HISTORY = 0.1
HISTORY_RATIO_WITH_HISTORY = 0.2



#-------------------------------------------------------
#-----prompt for entities and timestamps extarction-----
#-------------------------------------------------------

class ContextInfoExtractionResult(BaseModel):
    extracted_entities: Optional[List[str]]
    downer_border_event_name: Optional[str]
    upper_border_event_name: Optional[str]

context_info_parser = PydanticOutputParser(pydantic_object=ContextInfoExtractionResult)

prompt_context_info_en = ChatPromptTemplate.from_messages([
    ("system", retrieval_prompts.SYSTEM_PROMPT_TIMESPAMPS_EN),
    ("human",
        "Text:\n{query_text}\n\n"
        "List of extracted events and their names:\n{events_names}\n\n"
        "{format_instructions}")
]).partial(format_instructions=context_info_parser.get_format_instructions())

prompt_context_info_ru = ChatPromptTemplate.from_messages([
    ("system", retrieval_prompts.SYSTEM_PROMPT_TIMESPAMPS_RU),
    ("human",
        "Text:\n{query_text}\n\n"
        "List of extracted events and their names:\n{events_names}\n\n"
        "{format_instructions}")
]).partial(format_instructions=context_info_parser.get_format_instructions())



#---------------------------
#-----history retrieval-----
#---------------------------

def extract_event_sequence(graph: KnowledgeGraph) -> Dict[str, int]:

    events = {n.id: n for n in graph.get_all_nodes() if n.type == "event"}

    followers = defaultdict(set)
    indegree = {eid: 0 for eid in events}
    for edge in graph.get_all_edges():
        if edge.source not in events or edge.target not in events:
            continue
        src = None
        dst = None
        if edge.relation in ("precedes", "предшествует"):
            src = edge.source
            dst = edge.target
        elif edge.relation in ("follows", "следует за", "следует"):
            src = edge.target
            dst = edge.source
        if src is None:
            continue
        if dst not in followers[src]:
            followers[src].add(dst)
            indegree[dst] += 1

    def chunk_id(eid):
        chunks = events[eid].chunk_id
        if not chunks:
            return float("inf")
        return min(chunks)
    
    available = [eid for eid in events if indegree[eid] == 0]
    result = {}
    stage = 0
    processed = 0
    while available:
        grouped = defaultdict(list)
        for eid in available:
            grouped[chunk_id(eid)].append(eid)
        next_available = []
        for chunk in sorted(grouped):
            same_chunk_events = grouped[chunk]
            for eid in same_chunk_events:
                result[eid] = stage
                processed += 1
            for eid in same_chunk_events:
                for nxt in followers[eid]:
                    indegree[nxt] -= 1
                    if indegree[nxt] == 0:
                        next_available.append(nxt)
            stage += 1
        available = next_available

    if processed != len(events):
        return {}
    return result


def extract_world_history(
        events_sequence: Dict[str, int], #event_id: number in sequence
        graph: KnowledgeGraph,
        max_tokens: int = 512
) -> str:
    
    result_history_dict = {} # { event_id : ("event_name. event_time. event_description.", token_amount) }
    event_ids = [event_id for event_id in events_sequence.keys()]
    
    for event_id in event_ids:
        event = graph.get_node_by_id(event_id)
        text = f"{event.name}. "
        if event.base_attributes.get("time"):
            text += f"{event.base_attributes.get('time')}. "
        text += f"{event.base_description}"
        tokens = estimate_tokens(text)
        result_history_dict[event_id] = (text, tokens)
    
    result = "HISTORY:\n\n"
    total_tokens = sum(t for _, t in result_history_dict.values())
    if total_tokens <= max_tokens:
        for event_description in result_history_dict.values():
            text = event_description[0]
            result += text + "\n"
        return result
    
    middle_event_ids = event_ids[1:-1]
    while total_tokens > max_tokens and middle_event_ids:
        remove_id = random.choice(middle_event_ids)
        middle_event_ids.remove(remove_id)
        total_tokens -= result_history_dict[remove_id][1]
    final_ids = [event_ids[0]] + middle_event_ids + [event_ids[-1]]
    for event in final_ids:
        text = result_history_dict[event_id][0]
        result += text + "\n"
    return result



#----------------------------------------
#-----retrieval base pipeline stages-----
#----------------------------------------

def retrieve_similar_nodes(
        graph: KnowledgeGraph,
        text: str,
        embedding_model: Embeddings,
        amount: int = 10,
        threshold: float = 0.55,

        use_filtration: bool = False,
        upper_border_event_id: str = "",
        events_sequence: list = []
    ) -> List[Tuple[Node, float]]:

    vector_db = graph.get_vector_db()
    query_emb = np.array(embedding_model.embed_query(text))
    results = vector_db.search(query_emb, amount)
    candidates = []
    candidate_nodes = []
    for result in results:
        if result.get("distance", 0.0) >= threshold:
            if(result.get("metadata").get("type") == "edge"):
                edge = graph.get_edge_by_id(result.get("id"))
                if (use_filtration):
                    max_time = max(events_sequence.values())+1 if events_sequence else 0
                    min_time = 0
                    reference_time = events_sequence.get(upper_border_event_id, max_time) if upper_border_event_id else max_time
                    if edge:
                        edge_first_appearance = events_sequence.get(edge.time_start_event, min_time) if edge.time_start_event else min_time
                        if edge_first_appearance < reference_time:
                            node1 = graph.get_node_by_id(result.get("metadata").get("source"))
                            node2 = graph.get_node_by_id(result.get("metadata").get("target"))
                            if node1 and not node1.id in candidate_nodes:
                                candidates.append((node1, result.get("distance", 0.0)))
                                candidate_nodes.append(node1.id)
                            if node2 and not node2.id in candidate_nodes:
                                candidates.append((node2, result.get("distance", 0.0)))
                                candidate_nodes.append(node2.id)
                else:
                    node1 = graph.get_node_by_id(result.get("metadata").get("source"))
                    node2 = graph.get_node_by_id(result.get("metadata").get("target"))
                    if node1 and not node1.id in candidate_nodes:
                        candidates.append((node1, result.get("distance", 0.0)))
                        candidate_nodes.append(node1.id)
                    if node2 and not node2.id in candidate_nodes:
                        candidates.append((node2, result.get("distance", 0.0)))
                        candidate_nodes.append(node2.id)
            else:         
                node = graph.get_node_by_id(result.get("metadata").get("original_id"))
                if node and node.id in candidate_nodes:
                    candidates.append((node, result.get("distance", 0.0) + 2.0))
                    candidate_nodes.append(node.id)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates

def find_paths(
        graph: KnowledgeGraph,
        start_id: str,
        end_id: str,

        should_filter_by_time: bool = False,
        event_sequence: Dict[str, int] = {}, #event_id: number in sequence
        upper_border_event_id: str = None,

        alpha: float=0.7,
        threshold: float=0.01,
        max_depth: int=5
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
        outgoing_edges = [e for e in graph.get_all_edges() if e.source == current]
        
        if (should_filter_by_time):
            outgoing_edges_filtered = []
            max_time = max(event_sequence.values()) if event_sequence else 0
            min_time = 0
            reference_time = event_sequence.get(upper_border_event_id, max_time) if upper_border_event_id else max_time
            for edge in outgoing_edges:
                edge_first_appearance = event_sequence.get(edge.time_start_event, min_time) if edge.time_start_event else min_time
                if edge_first_appearance < reference_time:
                    outgoing_edges_filtered.append(edge)
            outgoing_edges = outgoing_edges_filtered
        
        out_degree = len(outgoing_edges)
        for edge in outgoing_edges:
            next_node = edge.target
            if any(n == next_node for n in path):
                continue
            S_next = alpha * (S_current / max(out_degree, 1))
            if S_next < threshold:
                continue
            new_path = path + [next_node]
            new_edges = edges + [edge.id]
            queue.append((next_node, new_path, new_edges, S_next))
    results.sort(key=lambda x: x[2], reverse=True)
    return results



#-------------------------------------
#-----context retrieval functions-----
#-------------------------------------

def form_context_without_llm(
        query: str,
        graph: KnowledgeGraph,
        embedding_model: Embeddings,
        add_history: bool=False,
        max_tokens: int = 1024,
        language: str = "en"
    ) -> str:

    events_sequence = extract_event_sequence(graph) #event_id: number in sequence

    result_nodes_dict = {} # { node_id : ("node_name. node_description", token_amount) }
    result_edges_dict = {} # { (source_id, target_id) : ("name_source - relation - name_target", token_amount) }
    result_paths_dict = {} # { (source_id, target_id, number) : ("name_node_start - relation - ... - relation - name_node_target", token_amount) }

    update_entities_from_graph(graph=graph)
    extracted_entities_names = extract_mentioned_entities(query=query, language=language)

    entry_nodes = []
    for entity_name in extracted_entities_names:
        if graph.get_node_by_name(entity_name):
            entry_nodes.append((graph.get_node_by_name(entity_name), 10.0))
            continue
        else:
            entity_name_entry = retrieve_similar_nodes(graph=graph, text=entity_name, embedding_model=embedding_model, threshold=0.75)
            if len(entity_name_entry) > 0:
                entry_nodes.extend(entity_name_entry)
    if len(entry_nodes) == 0:
        entry_nodes = retrieve_similar_nodes(graph=graph, text=query, embedding_model=embedding_model, amount=10, threshold=0.0)
    result_nodes = entry_nodes.copy() 
    for entry_node, resource in entry_nodes:
        neighbours = graph.get_neighbours_of_node(entry_node.id)
        for neighbour in neighbours:
            if not any(node[0] == neighbour for node in result_nodes):
                edges = [edge for edge in graph.get_all_edges() if edge.target == neighbour.id]
                edges = [edge for edge in edges if edge.source == entry_node.id]
                if not edges:
                    continue
                edges.sort(key=lambda x: x.weight)
                weight = edges[0].weight
                result_nodes.append((neighbour, weight))
                for edge in edges:
                    text = graph.get_node_by_id(edge.source).name + " " + edge.relation + " " + graph.get_node_by_id(edge.target).name + ". " + edge.description
                    tokens = estimate_tokens(text)
                    result_edges_dict[(edge.source, edge.target)] = (text, tokens)

    paths_nodes = []
    if len(entry_nodes) > 1:
        for a, b in combinations(entry_nodes, 2):
            paths_nodes.append(find_paths(graph, a[0].id, b[0].id))

    seen = set()
    all_paths = [
        p for paths in paths_nodes 
        for p in paths 
        if len(p[0]) > 2 and (tuple(p[0]) not in seen and not seen.add(tuple(p[0])))
    ]

    for path in all_paths:
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
            if i > 0 and i < len(path_nodes)-1:
                pretty_parts.append(" who (what)")
            if i < len(path_edges):
                edge_id = path_edges[i]
                edge = graph.get_edge_by_id(edge_id)
                pretty_parts.append(f" - {edge.relation} - ")
        pretty_string = "".join(pretty_parts)
        result_paths_dict[(source_id, target_id, index)] = (pretty_string, score)

    result_nodes.sort(key=lambda x: x[1])
    for node, weight in result_nodes:
        text = f"{node.name}. {node.base_description}"
        tokens = estimate_tokens(text)
        result_nodes_dict[node.id] = (text, tokens)
    
    if (add_history):
        node_budget = int(max_tokens * NODE_RATIO_WITH_HISTORY)
        edge_budget = int(max_tokens * EDGE_RATIO_WITH_HISTORY)
        path_budget = int(max_tokens * PATH_RATIO_WITH_HISTORY)
    else:
        node_budget = int(max_tokens * NODE_RATIO_WITHOUT_HISTORY)
        edge_budget = int(max_tokens * EDGE_RATIO_WITHOUT_HISTORY)
        path_budget = int(max_tokens * PATH_RATIO_WITHOUT_HISTORY)

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
    
    result = ""

    if (add_history):
        history_max_tokens = int(max_tokens * HISTORY_RATIO_WITH_HISTORY)
        history = extract_world_history(events_sequence, graph, history_max_tokens)
        result += history

    result += "PATHS:\n\n"
    for (_, text) in selected_paths:
        result += text + "\n"
    result += "\nEDGES:\n\n"
    for (_, text) in selected_edges:
        result += text + "\n"
    result += "\nNODES:\n\n"
    for node_id in selected_nodes:
        text = result_nodes_dict[node_id][0]
        result += text + "\n"

    return result 


def form_context_with_llm(
        query: str,
        graph: KnowledgeGraph,
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        language: str = "en",
        add_history: bool=False,
        max_tokens: int = 1024
    ) -> str:

    if language == "ru":
        chain_timestamps = prompt_context_info_ru | llm | clean_json | context_info_parser
    else:
        chain_timestamps = prompt_context_info_en | llm | clean_json | context_info_parser

    events_sequence = extract_event_sequence(graph) #event_id: number in sequence
    events_nodes = []
    events_names = []
    for event in events_sequence.keys():
        event_node = graph.get_node_by_id(event)
        events_nodes.append(event_node)
        events_names.append(f"{event_node.name}. {event_node.base_description}")

    update_entities_from_graph(graph=graph)
    context_info = chain_timestamps.invoke({ "query_text": query, "events_names": events_names })

    context_info.downer_border_event_name = clean_string(context_info.downer_border_event_name) if context_info.downer_border_event_name else None
    context_info.upper_border_event_name = clean_string(context_info.upper_border_event_name) if context_info.upper_border_event_name else None

    node_downer_event = graph.get_node_by_name(context_info.downer_border_event_name) if context_info.downer_border_event_name else None
    downer_border_event_id = node_downer_event.id if node_downer_event else None
    node_upper_event = graph.get_node_by_name(context_info.upper_border_event_name) if context_info.upper_border_event_name else None
    upper_border_event_id = node_upper_event.id if node_upper_event else None
    extracted_entities_names = context_info.extracted_entities if context_info.extracted_entities else []
    extracted_entities_names = extract_mentioned_entities(query=query, already_extracted_entities=extracted_entities_names, language=language)
    
    result_nodes_dict = {} # { node_id : ("node_name. node_description", token_amount) }
    result_edges_dict = {} # { (source_id, target_id) : ("name_source - relation - name_target", token_amount) }
    result_paths_dict = {} # { (source_id, target_id, number) : ("name_node_start - relation - ... - relation - name_node_target", token_amount) }

    entry_nodes = []
    for entity_name in extracted_entities_names:
        if graph.get_node_by_name(entity_name):
            entry_nodes.append((graph.get_node_by_name(entity_name), 1.0))
            continue
        else:
            entity_name_entry = retrieve_similar_nodes(graph=graph, text=entity_name, embedding_model=embedding_model, threshold=0.75)
            if len(entity_name_entry) > 0:
                entry_nodes.extend(entity_name_entry)
    if len(entry_nodes) == 0:
        entry_nodes = retrieve_similar_nodes(graph=graph, text=query, embedding_model=embedding_model, amount=10, threshold=0.0, use_filtration=True, upper_border_event_id=upper_border_event_id, events_sequence=events_sequence)

    result_nodes = entry_nodes.copy()
    for entry_node, resource in entry_nodes:
        neighbours = graph.get_neighbours_of_node(entry_node.id)
        for neighbour in neighbours:
            if not any(node[0] == neighbour for node in result_nodes):
                edges = [edge for edge in graph.get_all_edges() if edge.target == neighbour.id]
                edges = [edge for edge in edges if edge.source == entry_node.id]
                edges.sort(key=lambda x: x.weight)
                max_time = max(events_sequence.values())+1 if events_sequence else 0
                min_time = 0

                edges_filtered = []
                reference_time = events_sequence.get(upper_border_event_id, max_time) if upper_border_event_id else max_time
                for edge in edges:
                    edge_first_appearance = events_sequence.get(edge.time_start_event, min_time) if edge.time_start_event else min_time
                    if edge_first_appearance < reference_time:
                        edges_filtered.append(edge)
                edges = edges_filtered
                if (edges):
                    weight = edges[0].weight
                    result_nodes.append((neighbour, weight))
                    reference_time = events_sequence.get(downer_border_event_id, min_time) if downer_border_event_id else min_time
                    for edge in edges:
                        edge_last_appearance = events_sequence.get(edge.time_end_event, max_time) if edge.time_end_event else max_time
                        if (edge_last_appearance < reference_time):
                            text = graph.get_node_by_id(edge.source).name + " " + edge.relation + " " + graph.get_node_by_id(edge.target).name + " (before " + context_info.downer_border_event_name + ")" + ". " + edge.description
                        else:
                            text = graph.get_node_by_id(edge.source).name + " " + edge.relation + " " + graph.get_node_by_id(edge.target).name + ". " + edge.description
                        tokens = estimate_tokens(text)
                        result_edges_dict[(edge.source, edge.target)] = (text, tokens)

    paths_nodes = []
    if len(entry_nodes) > 1:
        for a, b in combinations(entry_nodes, 2):
            paths_nodes.append(find_paths(graph, a[0].id, b[0].id, True, events_sequence, upper_border_event_id))

    seen = set()
    all_paths = [
        p for paths in paths_nodes 
        for p in paths 
        if len(p[0]) > 2 and (tuple(p[0]) not in seen and not seen.add(tuple(p[0])))
    ]

    for path in all_paths:
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
            if i > 0 and i < len(path_nodes)-1:
                pretty_parts.append(" who (what)")
            if i < len(path_edges):
                edge_id = path_edges[i]
                edge = graph.get_edge_by_id(edge_id)

                reference_time = reference_time = events_sequence.get(downer_border_event_id, 0) if downer_border_event_id else 0
                max_time = max(events_sequence.values()) if events_sequence else 0

                edge_last_appearance = events_sequence.get(edge.time_end_event, max_time) if edge.time_end_event else max_time
                if (edge_last_appearance < reference_time):
                    pretty_parts.append(f" - {edge.relation} (before {context_info.downer_border_event_name}) - ")
                else:
                    pretty_parts.append(f" - {edge.relation} - ")
        pretty_string = "".join(pretty_parts)
        result_paths_dict[(source_id, target_id, index)] = (pretty_string, score)

    result_nodes.sort(key=lambda x: x[1])
    for node, weight in result_nodes:
        text = f"{node.name}."
        filtered_node = filter_node_states_by_time(node, events_sequence, downer_border_event_id, upper_border_event_id)
        if len(filtered_node.states) > 0:
            for state in filtered_node.states:
                text = text + f" {state.current_description}"
        else:
            text = text + f" {filtered_node.base_description}"
        tokens = estimate_tokens(text)
        result_nodes_dict[node.id] = (text, tokens)
    
    if (add_history):
        node_budget = int(max_tokens * NODE_RATIO_WITH_HISTORY)
        edge_budget = int(max_tokens * EDGE_RATIO_WITH_HISTORY)
        path_budget = int(max_tokens * PATH_RATIO_WITH_HISTORY)
    else:
        node_budget = int(max_tokens * NODE_RATIO_WITHOUT_HISTORY)
        edge_budget = int(max_tokens * EDGE_RATIO_WITHOUT_HISTORY)
        path_budget = int(max_tokens * PATH_RATIO_WITHOUT_HISTORY)

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
    
    result = ""

    if (add_history):
        history_max_tokens = int(max_tokens * HISTORY_RATIO_WITH_HISTORY)
        history = extract_world_history(events_sequence, graph, history_max_tokens)
        result += history

    result += "PATHS:\n\n"
    for (_, text) in selected_paths:
        result += text + "\n"

    result += "\nEDGES:\n\n"
    for (_, text) in selected_edges:
        result += text + "\n"

    result += "\nNODES:\n\n"
    for node_id in selected_nodes:
        text = result_nodes_dict[node_id][0]
        result += text + "\n"

    return result 