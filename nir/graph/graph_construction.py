#All stuff with graph extracting and updating is here



#--------------------------
#---------imports----------
#--------------------------

import re, json
import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from copy import deepcopy
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from fastcoref import FCoref

from nir.graph.graph_parser import SafePydanticParser, normalize_events_subgraph, normalize_graph_extraction_result, normalize_merged_node
from nir.graph.graph_structures import InputEdge, InputNode, MergedNode, Node, Edge, EventImpact, State, EventsSubgraph, GraphExtractionResult
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.prompts import extraction_prompts
from nir.embedding.vector_store import VectorStore



#--------------------------
#-----additional stuff-----
#--------------------------

def get_next_chunk_id(graph: KnowledgeGraph = None) -> int:
    if graph == None:
        return 0
    all_nodes = graph.get_all_nodes()
    all_edges = graph.get_all_edges()
    max_node_id = max([max(n.chunk_id) if n.chunk_id else 0 for n in all_nodes], default=-1)
    max_edge_id = max([e.chunk_id if isinstance(e.chunk_id, int) else max(e.chunk_id, default=0) for e in all_edges], default=-1)
    return max(max_node_id, max_edge_id) + 1

def create_id(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Zа-яА-Я\s]", "", name)
    return re.sub(r"\s+", "_", cleaned.strip()).lower()

def cosine_sim(text1: str, text2: str, embedding_model: Embeddings) -> float:
    embedding1 = np.array(embedding_model.embed_query(text1))
    embedding2 = np.array(embedding_model.embed_query(text2))
    return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

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
    codeblock_match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if codeblock_match:
        possible_json = codeblock_match.group(1).strip()
        cleaned = remove_comments(possible_json)
        return cleaned

    balanced = extract_last_json(text)
    if balanced:
        try:
            json.loads(balanced)
            cleaned = remove_comments(balanced)
            return cleaned
        except json.JSONDecodeError:
            pass

    cleaned = re.sub(r"^[^{]+", "", text)
    cleaned = re.sub(r"[^}]+$", "", cleaned)
    cleaned = remove_comments(cleaned)
    cleaned = re.sub(r'(":\s*"[^"]*")\s*\([^)]*\)', r'\1', cleaned)
    return cleaned

def resolve_coreference(chunk_text: str) -> List[List[str]]:
    corefres_model = FCoref(device='cuda:0')
    entities = corefres_model.predict(texts=[chunk_text])
    return entities[0].get_clusters()

def create_input_node(full_node: Node) -> InputNode:
    input_node = InputNode(
        id=full_node.id,
        name=full_node.name,
        base_description=full_node.base_description,
        base_attributes=full_node.base_attributes,
        states=full_node.states
    )
    return input_node

def create_input_edge(full_edge: Edge) -> InputEdge:
    input_edge = InputEdge(
        id=full_edge.id,
        source=full_edge.source,
        target=full_edge.target,
        relation=full_edge.relation,
        description=full_edge.description,
        time_start_event=full_edge.time_start_event,
        time_end_event=full_edge.time_end_event
    )
    return input_edge

def safe_invoke_chain(chain: Runnable, inputs: Dict[str, Any], max_retries: int = 1) -> Optional[Any]:
    @retry(
        stop=stop_after_attempt(max_retries + 1),
        retry=retry_if_exception_type((ValueError, json.JSONDecodeError, Exception)),
        reraise=False
    )
    def _invoke():
        return chain.invoke(inputs)
    try:
        return _invoke()
    except Exception as e:
        return None



#----------------------------------------------
#-----first stage of extraction (entities)-----
#----------------------------------------------

entities_parser = PydanticOutputParser(pydantic_object=GraphExtractionResult)
safe_entities_parser = SafePydanticParser(expected_structure=GraphExtractionResult, normalizer=normalize_graph_extraction_result)

prompt_entities = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_ENTITIES),
    ("human",
        "Text fragment:\n{chunk_text}\n\n"
        "Coreference clusters:\n{coreference_array}\n\n"
        "{format_instructions}")
]).partial(format_instructions=entities_parser.get_format_instructions())

#-----------------------------------------------------
#-----second stage of extraction (events' impact)-----
#-----------------------------------------------------

events_parser = PydanticOutputParser(pydantic_object=EventsSubgraph)
safe_events_parser = SafePydanticParser(expected_structure=EventsSubgraph, normalizer=normalize_events_subgraph)

prompt_events = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_EVENTS),
    ("human",
        "Text:\n{chunk_text}\n\n"
        "Events:\n{events_list}\n\n"
        "Entities:\n{entities_list}\n\n"
        "Relations:\n{edges_list}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=events_parser.get_format_instructions())

#--------------------------------------------------
#-----semistages of extraction (merging nodes)-----
#--------------------------------------------------

merged_nodes_parser = PydanticOutputParser(pydantic_object=MergedNode)
safe_merged_nodes_parser = SafePydanticParser(expected_structure=MergedNode, normalizer=normalize_merged_node)

prompt_merging = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_MERGING),
    ("human",
        "Node A:\n{node_a_json}\n\n"
        "Node B:\n{node_b_json}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=merged_nodes_parser.get_format_instructions())



#--------------------------------------------------
#---------extracting and updating functions--------
#--------------------------------------------------

def merge_similar_nodes(
        nodes: List[Node], 
        edges: List[Edge], 
        llm: BaseLanguageModel, 
        embedding_model: Embeddings,
        preserve_all_data: bool = True,
        similarity_threshold: float = 0.85
    ) -> Tuple[Dict[str, Node], Dict[str, Edge]]:  
    
    chain_merging = prompt_merging | llm | clean_json | merged_nodes_parser
    safe_chain_merging = prompt_merging | llm | safe_merged_nodes_parser.parse

    if not nodes:
        return List(nodes={}, edges={})
    
    merged_nodes = []
    id_map = {}
    for node in nodes:
        is_node_merged = False
        node_i_description = f"{node.name}. {node.base_description}"
        for idx, existing in enumerate(merged_nodes):
            node_j_description = f"{existing.name}. {existing.base_description}"
            if node.type == existing.type:
                sim = cosine_sim(node_i_description, node_j_description, embedding_model)                
                if sim >= similarity_threshold:
                   
                    node_for_llm = MergedNode(
                        name=node.name,
                        base_description=node.base_description,
                        base_attributes=node.base_attributes
                    )

                    existing_for_llm = MergedNode(
                        name=node.name,
                        base_description=node.base_description,
                        base_attributes=node.base_attributes
                    )

                    if preserve_all_data:
                        merged_node: MergedNode = safe_chain_merging.invoke({
                            "node_a_json": existing_for_llm.model_dump_json(),
                            "node_b_json": node_for_llm.model_dump_json()
                        }) 
                    else:
                        merged_node: MergedNode = safe_invoke_chain(
                            chaon=chain_merging, 
                            inputs={
                                "node_a_json": existing_for_llm.model_dump_json(),
                                "node_b_json": node_for_llm.model_dump_json()
                        })
                        if merged_node is None:
                            break
                    
                    if (merged_node.name != ""):
                        merged_node_to_graph = Node(
                            id=existing.id,
                            name=merged_node.name,
                            type=existing.type,
                            base_description=merged_node.base_description,
                            base_attributes=merged_node.base_attributes,
                            states=existing.states + node.states,
                            chunk_id=list(set(existing.chunk_id + node.chunk_id))
                        )

                        merged_nodes[idx] = merged_node_to_graph
                        id_map[node.id] = merged_node_to_graph.id
                        is_node_merged = True
                        break

        if not is_node_merged:
            merged_nodes.append(deepcopy(node))
            id_map[node.id] = node.id
    
    merged_nodes_dict = {n.id: n for n in merged_nodes}
    
    merged_edges_dict = {}
    for edge in edges:
        if edge.source in id_map.keys() and edge.target in id_map.keys():
            merged_edge = deepcopy(edge)
            merged_edge.source = id_map[edge.source]
            merged_edge.target = id_map[edge.target] 
            merged_edges_dict[merged_edge.id] = merged_edge
    return merged_nodes_dict, merged_edges_dict

def extract_entities(
        chunks: List[Document], 
        llm: BaseLanguageModel,
        preserve_all_data: bool = True
    ) -> Tuple[Dict[str, Node], Dict[str, Edge]]:
    
    all_nodes: Dict[str, Node] = {}
    all_edges: Dict[str, Edge] = {}

    chain_entities = prompt_entities | llm | clean_json | entities_parser
    safe_chain_entities = prompt_entities | llm | safe_entities_parser.parse

    for idx, chunk in enumerate(chunks):
        
        print(f"[Chunk {idx+1}/{len(chunks)}] Extracting nodes and edges.") #DEBUGGING
        
        coreference_array = resolve_coreference(chunk.page_content)

        if preserve_all_data:
            result: GraphExtractionResult = safe_chain_entities.invoke({
                "chunk_text": chunk.page_content,
                "coreference_array": coreference_array
            })
        else:
            result: GraphExtractionResult = safe_invoke_chain(
                chain=chain_entities, 
                inputs={
                    "chunk_text": chunk.page_content,
                    "coreference_array": coreference_array
            })
            if result == None:
                break

        names_to_ids: Dict[str, str] = {}
        for extracted_node in result.nodes:
            node_id = create_id(extracted_node.name)
            names_to_ids[extracted_node.name] = node_id
            if node_id not in all_nodes:
                node = Node(
                    id=node_id,
                    name=extracted_node.name,
                    type=extracted_node.type,
                    base_description=extracted_node.base_description,
                    base_attributes=extracted_node.base_attributes,
                    states = [],
                    chunk_id = [chunk.metadata["chunk_id"]]
                )
                all_nodes[node_id] = node
        
        for extracted_edge in result.edges:
            edge_id_from1to2 = create_id(f"{extracted_edge.node1} {extracted_edge.relation_from1to2} {extracted_edge.node2}")
            edge_id_from2to1 = create_id(f"{extracted_edge.node2} {extracted_edge.relation_from2to1} {extracted_edge.node1}")
            print(extracted_edge)
            if edge_id_from1to2 not in all_edges:
                if extracted_edge.node1 != None and extracted_edge.node2 != None:
                    if extracted_edge.node1 in names_to_ids.keys() and extracted_edge.node2 in names_to_ids.keys():
                        
                        edge_1to2 = Edge (
                            id=edge_id_from1to2,
                            source=names_to_ids[extracted_edge.node1],
                            target=names_to_ids[extracted_edge.node2],
                            relation=extracted_edge.relation_from1to2,
                            description=extracted_edge.description,
                            weight=extracted_edge.weight,
                            time_start_event=None,
                            time_end_event=None,
                            chunk_id=chunk.metadata["chunk_id"]
                        )
                        all_edges[edge_id_from1to2] = edge_1to2
                        
                        edge_2to1 = Edge (
                            id=edge_id_from2to1,
                            source=names_to_ids[extracted_edge.node2],
                            target=names_to_ids[extracted_edge.node1],
                            relation=extracted_edge.relation_from2to1,
                            description=extracted_edge.description,
                            weight=extracted_edge.weight,
                            time_start_event=None,
                            time_end_event=None,
                            chunk_id=chunk.metadata["chunk_id"]
                        )
                        all_edges[edge_id_from2to1] = edge_2to1

    print(f"Extracted {len(all_nodes)} nodes and {len(all_edges)} edges.") #DEBUGGING

    return all_nodes, all_edges

def apply_event_impact_on_graph(
        graph: KnowledgeGraph, 
        impact: EventImpact, 
        event: Node
    ) -> None:
    
    eid = event.id
    if (impact.affected_nodes):
        for node in impact.affected_nodes:
            existing = graph.get_node_by_id(node.id)
            if (existing):
                new_state = State(
                    sid=f"{eid}_{node.id}",
                    current_description=node.new_current_description,
                    current_attributes=node.new_current_attributes,
                    event_start=node.time_start_event,
                    event_end=node.time_end_event
                )
                graph.update_node_states(node.id, new_state)
        
    if (impact.affected_edges):
        for edge in impact.affected_edges:
            edge = graph.get_edge_by_id(edge.id)
            if (edge):
                if (edge.time_start_event and edge.time_end_event):
                    graph.update_edge_times(edge.id, edge.description, time_start_event=edge.time_start_event, time_end_event=edge.time_end_event)
                elif (edge.time_start_event):
                    graph.update_edge_times(edge.id, edge.description, time_start_event=edge.time_start_event)
                elif (edge.time_end_event):
                    graph.update_edge_times(edge.id, edge.description, time_end_event=edge.time_end_event)

def extract_events_impact(
        chunks: List[Document],
        nodes: List[Node],
        edges: List[Edge], 
        llm: BaseLanguageModel,
        preserve_all_data: bool = True
    ) -> List[EventImpact]:
    
    event_impacts_all = []
    chain_event = prompt_events | llm | clean_json | events_parser
    safe_chain_event = prompt_events | llm | safe_events_parser.parse

    for idx, chunk in enumerate(chunks):

        print(f"[Chunk {idx+1}/{len(chunks)}] Extracting events impact.") #DEBUGGING

        chunk_nodes = [node for node in nodes if idx in node.chunk_id]
        chunk_edges = [edge for edge in edges if idx == edge.chunk_id]
        event_names = [node.name for node in chunk_nodes if node.type == "event"]
        entities_nodes = [node for node in chunk_nodes if node.type != "event"]
        entities_input_nodes = [create_input_node(node) for node in entities_nodes]
        chunk_input_edges = [create_input_edge(edge) for edge in chunk_edges]

        if preserve_all_data:
            events_impacts: EventsSubgraph = safe_chain_event.invoke({
                "chunk_text": chunk.page_content,
                "events_list": event_names,
                "entities_list": entities_input_nodes,
                "edges_list": chunk_input_edges
            })
        else:
            result: GraphExtractionResult = safe_invoke_chain(
                chain=chain_event, 
                inputs={
                    "chunk_text": chunk.page_content,
                    "events_list": event_names,
                    "entities_list": entities_input_nodes,
                    "edges_list": chunk_input_edges
            })
            if result == None:
                break
        print(events_impacts.events_with_impact)
        if len(events_impacts.events_with_impact) > 0:
            event_impacts_all.append(events_impacts.events_with_impact[0])

    return event_impacts_all
    
def extract_graph(
        chunks: List[Document],       
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        graph_class = NetworkXGraph,
        preserve_all_data: bool = True
    ) -> KnowledgeGraph:
    
    graph = graph_class()
    nodes, edges = extract_entities(
        chunks=chunks, 
        llm=llm,
        preserve_all_data=preserve_all_data
    )
    all_nodes = [n for n in nodes.values()]
    all_edges = [e for e in edges.values()]

    merged_result = merge_similar_nodes(
        nodes=all_nodes, 
        edges=all_edges, 
        llm=llm, 
        embedding_model=embedding_model,
        preserve_all_data=preserve_all_data
    )
    for n in merged_result[0].values():
        graph.add_node(n)
    for e in merged_result[1].values():
        if e.source and e.target:
            graph.add_edge(e)

    print(f"Graph built with {len(merged_result[0])} nodes and {len(merged_result[1])} edges.") #DEBUGGING
    
    nodes_in_graph = graph.get_all_nodes()
    edges_in_graph = graph.get_all_edges()
    events_impacts = extract_events_impact(
        chunks=chunks, 
        nodes=nodes_in_graph, 
        edges=edges_in_graph, 
        llm=llm,
        preserve_all_data=preserve_all_data
    )  
    events_only = [node for node in nodes_in_graph if node.type == "event"]
    for event in events_impacts:
        for event_in_graph in events_only:
            if event_in_graph.name == event.event_name:
                apply_event_impact_on_graph(graph, event, event_in_graph)

    return graph

def update_graph(
        chunks: List[Document],       
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        graph: KnowledgeGraph,
        preserve_all_data: bool = True
    ) -> None:
    
    nodes, edges = extract_entities(
        chunks=chunks, 
        llm=llm,
        preserve_all_data=preserve_all_data
    )   
    
    all_nodes = graph.get_all_nodes()
    for n in nodes.values():
        all_nodes.append(n)
    
    all_edges = graph.get_all_edges()
    for e in edges.values():
        all_edges.append(e)

    merged_result = merge_similar_nodes(
        nodes=all_nodes, 
        edges=all_edges, 
        llm=llm, 
        embedding_model=embedding_model,
        preserve_all_data=preserve_all_data
    )
    
    for n in merged_result[0].values():
        if graph.get_node_by_id(n.id):
            if graph.get_node_by_id(n.id) != n:
                graph.update_node_full(n.id, n)
        else:
            graph.add_node(n)  
    for e in merged_result[1].values():
        if e.source and e.target:
            if graph.get_edge_by_id(e.id):
                if graph.get_edge_by_id(e.id) != e:
                    graph.update_edge_full(e.id, e)
            else:   
                graph.add_edge(e)

    nodes_in_graph = graph.get_all_nodes()
    edges_in_graph = graph.get_all_edges()
    events_impacts = extract_events_impact(
        chunks=chunks, 
        nodes=nodes_in_graph, 
        edges=edges_in_graph, 
        llm=llm,
        preserve_all_data=preserve_all_data
    )  
    
    events_only = [node for node in nodes_in_graph if node.type == "event"]
    for event in events_impacts:
        for event_in_graph in events_only:
            if event_in_graph.name == event.event_name:
                apply_event_impact_on_graph(graph, event, event_in_graph)



#-------------------------------------------------
#---------creating and updating embeddings--------
#-------------------------------------------------

def create_embeddings(
        graph: KnowledgeGraph, 
        vector_store: VectorStore, 
        embedding_model: Embeddings
    ) -> None:
    
    nodes = graph.get_all_nodes()
    edges = graph.get_all_edges()
    all_ids = []
    all_documents = []
    all_metadatas = []

    for node in nodes:
        text = node.name 
        all_ids.append(node.id)
        all_documents.append(text)
        all_metadatas.append({
            "type": "node",
            "node_type": node.type,
            "name": node.name
        })
    for edge in edges:
        text = f"{edge.source} {edge.relation} {edge.target}"
        all_ids.append(edge.id)
        all_documents.append(text)
        all_metadatas.append({
            "type": "edge",
            "relation": edge.relation,
            "source": edge.source,
            "target": edge.target
        })
    embeddings = embedding_model.embed_documents(all_documents)
    vector_store.add_embeddings(
        ids=all_ids,
        embeddings=embeddings,
        metadatas=all_metadatas,
        documents=all_documents
    )
    vector_store.persist()

def update_embeddings(
        graph: KnowledgeGraph,
        vector_store: VectorStore,
        embedding_model: Embeddings
    ) -> None:
    
    graph_ids = []
    graph_docs = []
    graph_metadatas = []
    for node in graph.get_all_nodes():
        graph_ids.append(node.id)
        graph_docs.append(node.name)
        graph_metadatas.append({
            "type": "node",
            "node_type": node.type,
            "name": node.name
        })   
    store_ids = set(vector_store.get_all_ids())
    graph_ids_set = set(graph_ids)

    removed_ids = list(store_ids - graph_ids_set)
    new_ids = list(graph_ids_set - store_ids)
    possible_updated_ids = list(graph_ids_set & store_ids)

    if removed_ids:
        vector_store.delete_embeddings(removed_ids)

    if new_ids:
        new_docs = []
        new_metadatas = []
        for nid in new_ids:
            idx = graph_ids.index(nid)
            new_docs.append(graph_docs[idx])
            new_metadatas.append(graph_metadatas[idx])
        new_embeddings = embedding_model.embed_documents(new_docs)

        vector_store.add_embeddings(
            ids=new_ids,
            embeddings=new_embeddings,
            metadatas=new_metadatas,
            documents=new_docs
        )

    updated_ids = []
    updated_docs = []
    updated_metadatas = []
    for nid in possible_updated_ids:
        idx = graph_ids.index(nid)
        graph_meta = graph_metadatas[idx]
        store_meta = vector_store.get_metadata(nid) or {}
        if graph_meta != store_meta:
            updated_ids.append(nid)
            updated_docs.append(graph_docs[idx])
            updated_metadatas.append(graph_meta)
    if updated_ids:
        updated_embeddings = embedding_model.embed_documents(updated_docs)
        vector_store.update_embeddings(
            ids=updated_ids,
            embeddings=updated_embeddings,
            metadatas=updated_metadatas,
            documents=updated_docs
        )

