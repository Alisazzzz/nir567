#All stuff with graph extraction is here


#-----imports-----

import re, json
import numpy as np
from typing import List, Dict, Tuple, Any
from pydantic import BaseModel, Field
from copy import deepcopy

from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from fastcoref import FCoref

from nir.graph.graph_structures import NodeType, Node, Edge, EventImpact, State, EventsSubgraph
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.prompts import extraction_prompts


#-----additional stuff-----

class GraphExtractionResult(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

def normalize_id(text: str) -> str:
    return text.lower().replace(" ", "_").strip()

def cosine_sim(text1: str, text2: str, embedding_model: Embeddings) -> float:
    embedding1 = np.array(embedding_model.embed_query(text1))
    embedding2 = np.array(embedding_model.embed_query(text2))
    return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

def clean_non_json(text: str) -> str:
    """
    Пытается вычистить мусор с начала или конца строки,
    оставив только JSON.
    """
    # Удаляем строки до '{'
    text = re.sub(r"^[^{]+", "", text)
    # Удаляем строки после '}'
    text = re.sub(r"[^}]+$", "", text)
    return text



def _remove_json_comments_preserving_strings(s: str) -> str:
    """
    Удаляет комментарии из строки JSON:
    - однострочные: // ... (до конца строки), # ...
    - многострочные: /* ... */
    При этом не удаляет такие последовательности, если они находятся внутри кавычек ("
    или '), и корректно обрабатывает экранирование \" и \\.
    Алгоритм — побайтовый проход с отслеживанием состояния (внутри строки / вне строки / в комментарии).
    """
    out_chars = []
    i = 0
    n = len(s)
    in_string = False      # True когда внутри "..." или '...'
    string_quote = ""      # " или '
    in_single_line_comment = False
    in_multi_line_comment = False
    while i < n:
        c = s[i]

        # --- конец однострочного комментария
        if in_single_line_comment:
            if c == "\n":
                in_single_line_comment = False
                out_chars.append(c)  # сохраним сам перенос
            i += 1
            continue

        # --- конец многострочного комментария
        if in_multi_line_comment:
            if c == "*" and i + 1 < n and s[i + 1] == "/":
                in_multi_line_comment = False
                i += 2
            else:
                i += 1
            continue

        # --- если мы внутри строки, просто копируем, обрабатываем escape
        if in_string:
            if c == "\\":
                # скопировать символ экранирования и следующий символ (если есть)
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

        # --- не в строке и не в комментарии: проверяем начало комментариев или строк
        if c == '"' or c == "'":
            in_string = True
            string_quote = c
            out_chars.append(c)
            i += 1
            continue

        # начало однострочного комментария //
        if c == "/" and i + 1 < n and s[i + 1] == "/":
            in_single_line_comment = True
            i += 2
            continue

        # начало многострочного комментария /* ... */
        if c == "/" and i + 1 < n and s[i + 1] == "*":
            in_multi_line_comment = True
            i += 2
            continue

        # начало "shebang"-style or hash comment: #
        # treat '#' as comment start if it's at line start (or preceded by whitespace/newline)
        if c == "#":
            # check previous characters to verify it's start-of-line-like
            prev = s[i - 1] if i - 1 >= 0 else "\n"
            if prev in {"\n", "\r", "\t", " ", ""}:
                in_single_line_comment = True
                i += 1
                continue
            else:
                # if inside token (rare), keep the '#'
                out_chars.append(c)
                i += 1
                continue

        # обычный символ — копируем
        out_chars.append(c)
        i += 1

    return "".join(out_chars)

def extract_json(text: str) -> str:
    """
    Извлекает JSON-строку из вывода модели, не парся её в dict.
    """

    # --- 1. code block: ```json ... ```
    codeblock_match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if codeblock_match:
        possible_json = codeblock_match.group(1).strip()
        return possible_json

    # --- 2. любой блок { ... }
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        possible_json = brace_match.group(0)
        # Проверим, что это действительно JSON
        try:
            json.loads(possible_json)
            return possible_json
        except json.JSONDecodeError:
            pass  # попробуем ниже

    # --- 3. fallback: чистка мусора
    cleaned = clean_non_json(text)
    cleaned = _remove_json_comments_preserving_strings(cleaned)
    print("CLEANED JSON: " + cleaned)
    print("-------------------")
    return cleaned


def clean_chunk_id(json_str: str) -> str:
    """
    Принимает строку с JSON, исправляет поля chunk_id:
    - Для узлов: гарантируем, что chunk_id = список целых чисел
    - Для ребер: гарантируем, что chunk_id = целое число
    Возвращает строку с исправленным JSON.
    """
    # --- Попытка загрузить JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("Невозможно распарсить JSON из строки")

    # --- Исправляем chunk_id у nodes
    if "nodes" in data:
        for node in data["nodes"]:
            if "chunk_id" in node:
                # Всегда делаем список целых чисел
                if isinstance(node["chunk_id"], list):
                    node["chunk_id"] = [int(i) for i in node["chunk_id"]]
                else:
                    node["chunk_id"] = [int(node["chunk_id"])]
            else:
                node["chunk_id"] = [0]  # дефолтное значение

    # --- Исправляем chunk_id у edges
    if "edges" in data:
        for edge in data["edges"]:
            if "chunk_id" in edge:
                if isinstance(edge["chunk_id"], list):
                    # Берем первый элемент или 0, если пусто
                    edge["chunk_id"] = int(edge["chunk_id"][0]) if edge["chunk_id"] else 0
                else:
                    edge["chunk_id"] = int(edge["chunk_id"])
            else:
                edge["chunk_id"] = 0  # дефолтное значение

    # --- Возвращаем строку JSON
    return json.dumps(data, ensure_ascii=False, indent=2)


#-----first stage of extraction (entities)-----

entities_parser = PydanticOutputParser(pydantic_object=GraphExtractionResult)

prompt_hybrid = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_HYBRID),
    ("human",
        "Text fragment:\n{chunk_text}\n\n"
        "Coreference clusters:\n{coreference_array}\n\n"
        "{format_instructions}")
]).partial(format_instructions=entities_parser.get_format_instructions())

#-----second stage of extraction (events)-----

events_parser = PydanticOutputParser(pydantic_object=EventsSubgraph)

prompt_events = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_EVENTS),
    ("human",
        "Text:\n{chunk_text}\n\n"
        "Entities:\n{entities_json}\n\n"
        "Relations (edges):\n{edges_json}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=events_parser.get_format_instructions())

#-----semistages of extraction (merging nodes)-----

merged_nodes_parser = PydanticOutputParser(pydantic_object=Node)

prompt_merging = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_MERGING),
    ("human",
        "Node A:\n{node_a_json}\n\n"
        "Node B:\n{node_b_json}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=events_parser.get_format_instructions())

#-----graph extractor class-----

class GraphExtractor:
   
    def __init__(self,
                 llm: BaseLanguageModel,
                 embedding_model: Embeddings,
                 graph_class=NetworkXGraph,
                 similarity_threshold: float = 0.7,
                 coreference_list: List[List[str]] = []):
        self.llm = llm
        self.graph_class = graph_class
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.coreference_list = coreference_list
        self.chain_hybrid = prompt_hybrid | llm | extract_json | entities_parser
        self.chain_event = prompt_events | llm | extract_json | clean_chunk_id | events_parser
        self.merging_nodes = prompt_merging | llm | extract_json | merged_nodes_parser

    def resolve_coreference(self, chunk_text: str) -> List[List[str]]:
            corefres_model = FCoref(device='cuda:0')
            entities = corefres_model.predict(texts=[chunk_text])
            return entities[0].get_clusters()

    def _build_entity_map(self, entity_clusters: List[List[str]]) -> Dict[str, int]:
        entity_map = {}
        for idx, cluster in enumerate(entity_clusters):
            for name in cluster:
                entity_map[name] = idx
        return entity_map
    
    def _merge_similar_nodes(self, nodes:  Dict[str, Node], edges: Dict[str, Edge]) -> Tuple[Dict[str, Node], Dict[str, Edge]]:  
        if not nodes:
            return List(nodes={}, edges={})
        
        merged_nodes = []
        id_map = {}
        for node in nodes.values():
            is_node_merged = False
            node_i_description = f"{node.name}"
            for idx, existing in enumerate(merged_nodes):
                node_j_description = f"{existing.name}"
                sim = cosine_sim(node_i_description, node_j_description, self.embedding_model)                
                if sim >= self.similarity_threshold:
                    merged_node = self.merging_nodes.invoke({
                        "node_a_json": existing.model_dump_json(),
                        "node_b_json": node.model_dump_json()
                    })
                    if (merged_node.id != "None"):
                        merged_node.id = existing.id
                        merged_node.chunk_id = list(set(existing.chunk_id + node.chunk_id))
                        merged_nodes[idx] = merged_node
                        id_map[node.id] = merged_node.id
                        is_node_merged = True
                        break
            if not is_node_merged:
                merged_nodes.append(deepcopy(node))
                id_map[node.id] = node.id
        
        merged_nodes_dict = {n.id: n for n in merged_nodes}
        
        merged_edges_dict = {}
        for edge in edges.values():

            if edge.source in id_map and edge.target in id_map:
                merged_edge = deepcopy(edge)
                merged_edge.source = id_map[edge.source]       
                merged_edge.target = id_map[edge.target]
                merged_edges_dict[merged_edge.id] = merged_edge

        return merged_nodes_dict, merged_edges_dict

    def _extract_entities(self, chunks: List[Document], graph: KnowledgeGraph) -> None:
        all_nodes: Dict[str, Node] = {}
        all_edges: Dict[str, Edge] = {}

        for idx, chunk in enumerate(chunks):
            
            print(f"[Chunk {idx+1}/{len(chunks)}] Processing...") #DEBUGGING
            
            coreference_array = self.resolve_coreference(chunk.page_content)
            result: GraphExtractionResult = self.chain_hybrid.invoke({
                "chunk_text": chunk.page_content,
                "coreference_array": coreference_array
            })

            for node in result.nodes:
                norm_id = normalize_id(node.id)
                if norm_id not in all_nodes:
                    all_nodes[norm_id] = Node(**node.model_dump())
                    all_nodes[norm_id].chunk_id = [idx]
                    print(all_nodes[norm_id])
            
            for edge in result.edges:
                norm_id = normalize_id(edge.id)
                if norm_id not in all_edges:
                    if edge.target:
                        all_edges[norm_id] = edge
                        all_edges[norm_id].chunk_id = idx
                        print(all_edges[norm_id])
        
        merged_result = self._merge_similar_nodes(all_nodes, all_edges) #MERGING IS HERE

        for n in merged_result[0].values():
            graph.add_node(n)
        for e in merged_result[1].values():
            if e.source and e.target:
                graph.add_edge(e)

        print(f"Graph built with {len(all_nodes)} nodes and {len(all_edges)} edges.") #DEBUGGING

    def _apply_event_impacts(self, graph: KnowledgeGraph, events: List[EventImpact]) -> None:
        for idx, event in enumerate(events):
            eid = event.event_id
            for node in event.affected_nodes:
                node = graph.get_node_by_id(node.id)
                if (node):
                    new_description = node.description
                    new_state = State(
                        sid=f"{eid}_{node.id}",
                        attributes=node.attributes,
                        time_start=eid,
                        time_end=None
                    )
                    graph.update_node_state(node.id, new_description, new_state)

            for edge in event.affected_edges:
                node = graph.get_edge_by_id(node.id)
                if (edge):
                    new_description = edge.description
                    if (edge.time_start_event and edge.time_end_event):
                        graph.update_edge_times(edge.id, time_start_event=edge.time_start_event, time_end_event=edge.time_end_event)
                    elif (edge.time_start_event):
                        graph.update_edge_times(edge.id, time_start_event=edge.time_start_event)
                    elif (edge.time_end_event):
                        graph.update_edge_times(edge.id, time_end_event=edge.time_end_event)

    def _extract_events(self, chunks: List[Document], graph: KnowledgeGraph) -> None:
        for idx, chunk in enumerate(chunks):

            print(f"[Chunk {idx+1}/{len(chunks)}] Extracting events...") #DEBUGGING

            chunk_nodes = [node for node in graph.get_all_nodes() if idx in node.chunk_id]
            chunk_edges = [edge for edge in graph.get_all_edges() if idx == edge.chunk_id]
            
            entities_json = [n.model_dump_json() for n in chunk_nodes]
            edges_json = [e.model_dump_json() for e in chunk_edges]
            event_subgraph: EventsSubgraph = self.chain_event.invoke({
                "chunk_text": chunk.page_content,
                "entities_json": entities_json,
                "edges_json": edges_json
            })

            print(event_subgraph)

            for node in event_subgraph.nodes:
                node.chunk_id = [idx]
                if not any(n.id == node.id for n in graph.get_all_nodes()):
                    print(node)
                    graph.add_node(node)
            for edge in event_subgraph.edges:
                print(edge)
                edge.chunk_id = idx
                if not any(e.id == edge.id for e in graph.get_all_edges()):
                    graph.add_edge(edge)        
            self._apply_event_impacts(graph, event_subgraph.events_with_impact)

        print(f"Events integrated for chunk {idx}: "            #DEBUGGING
              f"{len(event_subgraph.nodes)} event nodes, "      #DEBUGGING
              f"{len(event_subgraph.edges)} event edges.")      #DEBUGGING
        
        print("Event extraction and integration complete.")     #DEBUGGING

    def extract_graph(self, chunks: List[Document]) -> KnowledgeGraph:
        graph = self.graph_class()
        self._extract_entities(chunks, graph)
        self._extract_events(chunks, graph)

        return graph
