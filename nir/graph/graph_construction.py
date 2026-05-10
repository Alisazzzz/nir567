#All stuff with graph extracting and updating is here



#--------------------------
#---------imports----------
#--------------------------

import re, json
import networkx as nx
import regex
import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from copy import deepcopy
import spacy
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from fastcoref import FCoref

from nir.embedding.vector_store_loader import VectorStoreInfo, create_vector_store
from nir.graph.graph_parser import SafePydanticParser, normalize_events_subgraph, normalize_graph_completion_result, normalize_graph_extraction_result, normalize_merged_in_graph_node, normalize_merged_node, normalize_merged_node_name, normalize_node_names_extraction_result
from nir.graph.graph_structures import GraphCompletionResult, InputEdge, InputNode, MergedInGraphNode, MergedNode, MergedNodeName, Node, Edge, EventImpact, NodeNamesExtractionResult, State, EventsSubgraph, GraphExtractionResult
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
    cleaned = regex.sub(r"[^\p{L}\p{N}\s]", "", name)
    return re.sub(r"\s+", "_", cleaned.strip()).lower()

def cosine_sim(text1: str, text2: str, embedding_model: Embeddings) -> float:
    q1 = f"{text1}"
    q2 = f"{text2}"
    embedding1 = np.array(embedding_model.embed_query(q1))
    embedding2 = np.array(embedding_model.embed_query(q2))
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

coref_models = {}
def resolve_coreference(chunk_text: str, language: str = "en") -> List[List[str]]:
    if language not in coref_models:
        nlp_model = "en_core_web_sm" if language == "en" else "ru_core_news_sm"
        coref_models[language] = FCoref(device='cuda:0', nlp=nlp_model)
    entities = coref_models[language].predict(texts=[chunk_text])
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

def get_next_unique_node_id(base_id: str, all_nodes: dict) -> str:
    max_num = 0
    prefix = f"{base_id}_"
    for key in all_nodes:
        if key.startswith(prefix):
            try:
                num = int(key.rsplit("_", 1)[1])
                if num > max_num:
                    max_num = num
            except (ValueError, IndexError):
                continue
    return f"{base_id}_{max_num + 1}"

def clean_entity_name(name: str) -> str:
    name = " ".join(name.split())
    name = name.strip('"\'')
    name = re.sub(r'^(a|an|the)\s+', '', name, flags=re.IGNORECASE)
    name = " ".join(name.split())
    return name if name else ""

def cluster_nodes_by_similarity(
    nodes: List[Node], 
    embedding_model: Embeddings, 
    threshold: float,
    use_description: bool = False
) -> List[List[Node]]:
    if not nodes:
        return []
    texts = []
    for n in nodes:
        if use_description:
            texts.append(f"{n.name} {n.base_description}")
        else:
            texts.append(n.name)

    embs = np.array(embedding_model.embed_documents(texts))

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10 
    embs_normalized = embs / norms
    sim_matrix = np.dot(embs_normalized, embs_normalized.T)
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if sim_matrix[i, j] >= threshold:
                if nodes[i].type == nodes[j].type or not nodes[i].type or not nodes[j].type:
                    G.add_edge(i, j)
    clusters = list(nx.connected_components(G))
    return [[nodes[i] for i in cluster] for cluster in clusters]

def get_chunk_text(chunks: List[Document], chunk_id: int) -> str:
    for chunk in chunks:
        if chunk.metadata.get("chunk_id") == chunk_id:
            return chunk.page_content
    return "No context available"

def split_cluster_by_confidence(
    cluster_indices: List[int],
    sim_matrix: np.ndarray,
    high_threshold: float,
    low_threshold: float,
) -> Tuple[List[int], List[int]]:
    if len(cluster_indices) == 1:
        return cluster_indices, []

    high_conf = [cluster_indices[0]]
    low_conf = []

    for idx in cluster_indices[1:]:
        sims_to_core = [sim_matrix[idx, j] for j in high_conf]
        avg_sim = float(np.mean(sims_to_core))
        if avg_sim >= high_threshold:
            high_conf.append(idx)
        else:
            low_conf.append(idx)

    return high_conf, low_conf


#----------------------------------------------
#-----first stage of extraction (entities)-----
#----------------------------------------------

entities_parser = PydanticOutputParser(pydantic_object=GraphExtractionResult)
safe_entities_parser = SafePydanticParser(expected_structure=GraphExtractionResult, normalizer=normalize_graph_extraction_result)

prompt_entities_en = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_ENTITIES_EN),
    ("human",
        "Text fragment:\n{chunk_text}\n\n"
        "Coreference clusters:\n{coreference_array}\n\n"
        "{format_instructions}")
]).partial(format_instructions=entities_parser.get_format_instructions())


prompt_entities_with_names_en = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_ENTITIES_WITH_NAMES_EN),
    ("human",
        "Text fragment:\n{chunk_text}\n\n"
        "Entities in chunk:\n{entities_array}\n\n"
        "{format_instructions}")
]).partial(format_instructions=entities_parser.get_format_instructions())


prompt_entities_ru = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_ENTITIES_RU),
    ("human",
        "Text fragment:\n{chunk_text}\n\n"
        "Coreference clusters:\n{coreference_array}\n\n"
        "{format_instructions}")
]).partial(format_instructions=entities_parser.get_format_instructions())

prompt_entities_with_names_ru = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_ENTITIES_WITH_NAMES_RU),
    ("human",
        "Text fragment:\n{chunk_text}\n\n"
        "Entities in chunk:\n{entities_array}\n\n"
        "{format_instructions}")
]).partial(format_instructions=entities_parser.get_format_instructions())


entities_names_parser = PydanticOutputParser(pydantic_object=NodeNamesExtractionResult)
safe_entities_names_parser = SafePydanticParser(expected_structure=NodeNamesExtractionResult, normalizer=normalize_node_names_extraction_result)

prompt_entities_names_en = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_ENTITIES_NAMES_EN),
    ("human",
        "Text fragment:\n{chunk_text}\n\n"
        "Coreference clusters:\n{coreference_array}\n\n"
        "{format_instructions}")
]).partial(format_instructions=entities_parser.get_format_instructions())

prompt_entities_names_ru = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_ENTITIES_NAMES_RU),
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

prompt_events_en = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_EVENTS_IMPACTS_EN),
    ("human",
        "Text:\n{chunk_text}\n\n"
        "Events:\n{events_list}\n\n"
        "Entities:\n{entities_list}\n\n"
        "Relations:\n{edges_list}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=events_parser.get_format_instructions())

prompt_events_ru = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_EVENTS_RU),
    ("human",
        "Text:\n{chunk_text}\n\n"
        "Events:\n{events_list}\n\n"
        "Entities:\n{entities_list}\n\n"
        "Relations:\n{edges_list}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=events_parser.get_format_instructions())



#-----------------------------------------------------
#-----semistages of extraction (completing graph)-----
#-----------------------------------------------------

completing_graph_parser = PydanticOutputParser(pydantic_object=GraphCompletionResult)
safe_completing_graph_parser = SafePydanticParser(expected_structure=GraphCompletionResult, normalizer=normalize_graph_completion_result)

prompt_completing_en = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_GRAPH_COMPLETION_EN),
    ("human",
        "Text:\n{chunk_text}\n\n"
        "Entities already extracted:\n{entities_list}\n\n"
        "Relations already extracted:\n{existing_relations}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=completing_graph_parser.get_format_instructions())

prompt_completing_ru = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_GRAPH_COMPLETION_RU),
    ("human",
        "Text:\n{chunk_text}\n\n"
        "Entities already extracted:\n{entities_list}\n\n"
        "Relations already extracted:\n{existing_relations}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=completing_graph_parser.get_format_instructions())



#--------------------------------------------------
#-----semistages of extraction (merging nodes)-----
#--------------------------------------------------

merged_nodes_parser = PydanticOutputParser(pydantic_object=MergedNode)
safe_merged_nodes_parser = SafePydanticParser(expected_structure=MergedNode, normalizer=normalize_merged_node)

prompt_merging_en = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_MERGING_EN),
    ("human",
        "Node A:\n{node_a_json}\n\n"
        "Node A context:\n{node_a_chunk}\n\n"
        "Node B:\n{node_b_json}\n\n"
        "Node A context:\n{node_b_chunk}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=merged_nodes_parser.get_format_instructions())

prompt_merging_ru = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_MERGING_RU),
    ("human",
        "Node A:\n{node_a_json}\n\n"
        "Node A context:\n{node_a_chunk}\n\n"
        "Node B:\n{node_b_json}\n\n"
        "Node A context:\n{node_b_chunk}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=merged_nodes_parser.get_format_instructions())

prompt_merging_cluster_en = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at Knowledge Graph deduplication. Merge these similar nodes into a single comprehensive node. {format_instructions}"),
    ("human", "Nodes to merge:\n{nodes_json}\n\nContexts:\n{contexts}")
]).partial(format_instructions=merged_nodes_parser.get_format_instructions())

prompt_merging_cluster_ru = ChatPromptTemplate.from_messages([
    ("system", "Ты - эксперт в избавлении от дубликатов вершин для графов знаний. Объедини эти одинаковые вершины в одну вершину, сохранив максимум информации. {format_instructions}"),
    ("human", "Вершины для объединения:\n{nodes_json}\n\Контексты:\n{contexts}")
]).partial(format_instructions=merged_nodes_parser.get_format_instructions())


merged_nodes_names_parser = PydanticOutputParser(pydantic_object=MergedNodeName)
safe_merged_nodes_names_parser = SafePydanticParser(expected_structure=MergedNodeName, normalizer=normalize_merged_node_name)

prompt_merging_names_en = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_MERGING_NAMES_EN),
    ("human",
        "Node A name:\n{node_a_name}\n\n"
        "Node A context:\n{node_a_context}\n\n"
        "Node B name:\n{node_b_name}\n\n"
        "Node B context:\n{node_b_context}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=merged_nodes_parser.get_format_instructions())

prompt_merging_names_ru = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_MERGING_NAMES_RU),
    ("human",
        "Node A name:\n{node_a_name}\n\n"
        "Node A context:\n{node_a_context}\n\n"
        "Node B name:\n{node_b_name}\n\n"
        "Node B context:\n{node_b_context}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=merged_nodes_parser.get_format_instructions())

prompt_merging_names_cluster_en = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at Knowledge Graph deduplication. Given these extracted entity names, provide the single best consolidated name. {format_instructions}"),
    ("human", "Names to merge:\n{names_json}\n\nContexts:\n{contexts}")
]).partial(format_instructions=merged_nodes_names_parser.get_format_instructions())

prompt_merging_names_cluster_ru = ChatPromptTemplate.from_messages([
    ("system", "Объедини приведенные имена вершин для одной сущности в одно имя, максимально отражающее сущность. {format_instructions}"),
    ("human", "Имена для объединения:\n{names_json}\n\Контекст:\n{contexts}")
]).partial(format_instructions=merged_nodes_names_parser.get_format_instructions())



#---------------------------------------
#-----editing graph (merging nodes)-----
#---------------------------------------

merged_in_graph_nodes_parser = PydanticOutputParser(pydantic_object=MergedInGraphNode)
safe_merged_in_graph_nodes_parser = SafePydanticParser(expected_structure=MergedInGraphNode, normalizer=normalize_merged_in_graph_node)

prompt_merging_in_graph_en = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_MERGING_IN_GRAPH_EN),
    ("human",
        "Node A:\n{node_a_json}\n\n"
        "Node B:\n{node_b_json}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=merged_in_graph_nodes_parser.get_format_instructions())

prompt_merging_in_graph_ru = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_MERGING_IN_GRAPH_RU),
    ("human",
        "Node A:\n{node_a_json}\n\n"
        "Node B:\n{node_b_json}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=merged_in_graph_nodes_parser.get_format_instructions())



#--------------------------------------------------
#---------extracting and updating functions--------
#--------------------------------------------------

def extract_entities_names(
        chunks: List[Document],
        llm: BaseLanguageModel,
        preserve_all_data: bool = True,
        language: str = "en"
) -> Dict[str, Node]:
    
    all_nodes: Dict[str, Node] = {}

    if language == "en":
        chain_entities_names = prompt_entities_names_en | llm | clean_json | entities_names_parser
        safe_chain_entities_names = prompt_entities_names_en | llm | safe_entities_names_parser.parse
    else:
        chain_entities_names = prompt_entities_names_ru | llm | clean_json | entities_names_parser
        safe_chain_entities_names = prompt_entities_names_ru | llm | safe_entities_names_parser.parse
    
    for idx, chunk in enumerate(chunks):
        
        print(f"[Chunk {idx+1}/{len(chunks)}] Extracting nodes names.") #DEBUGGING
        
        coreference_array = resolve_coreference(chunk.page_content, language=language)
        if preserve_all_data:
            result: NodeNamesExtractionResult = safe_chain_entities_names.invoke({
                "chunk_text": chunk.page_content,
                "coreference_array": coreference_array
            })
        else:
            result: NodeNamesExtractionResult = safe_invoke_chain(
                chain=chain_entities_names, 
                inputs={
                    "chunk_text": chunk.page_content,
                    "coreference_array": coreference_array
            })
            if result == None:
                continue

        for extracted_node in result.nodes:
            node_id = create_id(extracted_node.name)     
            if node_id not in all_nodes:
                node = Node(
                    id=node_id,
                    name=extracted_node.name,
                    type=extracted_node.type,
                    base_description="",
                    base_attributes={},
                    states = [],
                    chunk_id = [chunk.metadata["chunk_id"]]
                )
                all_nodes[node_id] = node
            else:
                node_id = get_next_unique_node_id(node_id, all_nodes)
                node = Node(
                    id=node_id,
                    name=extracted_node.name,
                    type=extracted_node.type,
                    base_description="",
                    base_attributes={},
                    states = [],
                    chunk_id = [chunk.metadata["chunk_id"]]
                )
                all_nodes[node_id] = node

    return all_nodes

def merge_similar_entities_names(
        chunks: List[Document],
        nodes: List[Node],
        llm: BaseLanguageModel, 
        embedding_model: Embeddings,
        preserve_all_data: bool = True,
        similarity_threshold: float = 0.85,
        language: str = "en"
) -> List[Node]:

    if not nodes:
        return []
        
    if language == "ru":
        chain_merging = prompt_merging_names_cluster_ru | llm | clean_json | merged_nodes_names_parser
        safe_chain_merging = prompt_merging_names_cluster_ru | llm | safe_merged_nodes_names_parser.parse
    else:
        chain_merging = prompt_merging_names_cluster_en | llm | clean_json | merged_nodes_names_parser
        safe_chain_merging = prompt_merging_names_cluster_en | llm | safe_merged_nodes_names_parser.parse
        
    exact_groups = {}
    for node in nodes:
        norm_name = clean_entity_name(node.name.lower())
        if norm_name not in exact_groups:
            exact_groups[norm_name] = []
        exact_groups[norm_name].append(node)
        
    unique_representatives = []
    for norm_name, group in exact_groups.items():
        if len(group) == 1:
            unique_representatives.append(group[0])
        else:
            base_node = deepcopy(group[0])
            for other in group[1:]:
                base_node.states.extend(other.states)
                base_node.chunk_id.extend(other.chunk_id)
            base_node.chunk_id = list(set(base_node.chunk_id))
            unique_representatives.append(base_node)
            
    clusters = cluster_nodes_by_similarity(unique_representatives, embedding_model, similarity_threshold, use_description=False)
    
    merged_nodes = []
    for cluster in clusters:
        if len(cluster) == 1:
            merged_nodes.append(cluster[0])
            continue
        print(f"Merging cluster of {len(cluster)} similar names: {[n.name for n in cluster]}")
        names_json = json.dumps([n.name for n in cluster], ensure_ascii=False)
        contexts = "\n".join([
            next((c.page_content for c in chunks if c.metadata["chunk_id"] == n.chunk_id[0]), "") 
            for n in cluster
        ])
        if preserve_all_data:
            merged_result = safe_chain_merging.invoke({"names_json": names_json, "contexts": contexts})
        else:
            merged_result = safe_invoke_chain(chain_merging, {"names_json": names_json, "contexts": contexts})   
        if merged_result and merged_result.name:
            base_node = deepcopy(cluster[0])
            base_node.name = merged_result.name
            for other in cluster[1:]:
                base_node.states.extend(other.states)
                base_node.chunk_id.extend(other.chunk_id)
            base_node.chunk_id = list(set(base_node.chunk_id))
            merged_nodes.append(base_node)
        else:
            merged_nodes.extend(cluster)
    return merged_nodes

def extract_graph_info(
        chunks: List[Document],
        nodes: List[Node],
        llm: BaseLanguageModel,
        preserve_all_data: bool = True,
        language: str = "en"
    ) -> Tuple[Dict[str, Node], Dict[str, Edge]]:

    all_nodes: Dict[str, Node] = {}
    all_edges: Dict[str, Edge] = {}
    nodes_in_work = {node.id : node for node in nodes}

    if language == "en":
        chain_entities = prompt_entities_with_names_en | llm | clean_json | entities_parser
        safe_chain_entities = prompt_entities_with_names_en | llm | safe_entities_parser.parse
    else:
        chain_entities = prompt_entities_with_names_ru | llm | clean_json | entities_parser
        safe_chain_entities = prompt_entities_with_names_ru | llm | safe_entities_parser.parse
    
    for idx, chunk in enumerate(chunks):
        
        print(f"[Chunk {idx+1}/{len(chunks)}] Extracting nodes info and edges.") #DEBUGGING
        
        nodes_in_chunk = [node for node in nodes_in_work.values() if chunk.metadata["chunk_id"] in node.chunk_id]
        if preserve_all_data:
            result: GraphExtractionResult = safe_chain_entities.invoke({
                "chunk_text": chunk.page_content,
                "entities_array": nodes_in_chunk
            })
        else:
            result: GraphExtractionResult = safe_invoke_chain(
                chain=chain_entities, 
                inputs={
                    "chunk_text": chunk.page_content,
                    "entities_array": nodes_in_chunk
            })
            if result == None:
                continue

        names_to_ids: Dict[str, str] = {}
        for extracted_node in result.nodes:
            node_id = create_id(extracted_node.name)
            if node_id not in nodes_in_work:
                node = Node(
                    id=node_id,
                    name=extracted_node.name,
                    type=extracted_node.type,
                    base_description=extracted_node.base_description,
                    base_attributes=extracted_node.base_attributes,
                    states = [],
                    chunk_id = [chunk.metadata["chunk_id"]]
                )
                nodes_in_work[node_id] = node
                names_to_ids[extracted_node.name] = node_id
            else:
                node = Node(
                    id=node_id,
                    name=nodes_in_work[node_id].name,
                    type=nodes_in_work[node_id].type,
                    base_description=extracted_node.base_description,
                    base_attributes=extracted_node.base_attributes,
                    states = [],
                    chunk_id = nodes_in_work[node_id].chunk_id
                )
                nodes_in_work[node_id] = node
                names_to_ids[extracted_node.name] = node_id
        all_nodes = deepcopy(nodes_in_work)

        for extracted_edge in result.edges:
            edge_id_from1to2 = create_id(f"{extracted_edge.node1} {extracted_edge.relation_from1to2} {extracted_edge.node2} edge")
            edge_id_from2to1 = create_id(f"{extracted_edge.node2} {extracted_edge.relation_from2to1} {extracted_edge.node1} edge")
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

def recursive_batch_merge(
    cluster: List[Node], 
    chunks: List[Document],
    chain, 
    safe_chain, 
    preserve_all_data: bool, 
    merge_type: str
) -> Node:

    if len(cluster) <= 1:
        return cluster[0]

    merged_results = []

    for i in range(0, len(cluster), 5):
        batch = cluster[i:i+5]
        if len(batch) == 1:
            merged_results.append(batch[0])
            continue
            
        print(f"Batch merging {len(batch)} nodes (type: {merge_type})...")
        if merge_type == "names":
            nodes_json = json.dumps([n.name for n in batch], ensure_ascii=False)
        else:
            nodes_json = json.dumps([
                MergedNode(name=n.name, base_description=n.base_description, base_attributes=n.base_attributes).model_dump() 
                for n in batch
            ], ensure_ascii=False)
            
        contexts = "\n\n".join([
            next((c.page_content for c in chunks if c.metadata["chunk_id"] == n.chunk_id[0]), "") 
            for n in batch
        ])
        if preserve_all_data:
            result = safe_chain.invoke(
                {"names_json": nodes_json, "contexts": contexts} if merge_type == "names" 
                else {"nodes_json": nodes_json, "contexts": contexts}
            )
        else:
            result = safe_invoke_chain(chain, 
                {"names_json": nodes_json, "contexts": contexts} if merge_type == "names" 
                else {"nodes_json": nodes_json, "contexts": contexts}
            )

        base_node = deepcopy(batch[0])
        if result and result.name:
            base_node.name = result.name
            if merge_type == "full":
                base_node.base_description = result.base_description
                base_node.base_attributes = result.base_attributes

        for other in batch[1:]:
            base_node.states.extend(other.states)
            base_node.chunk_id.extend(other.chunk_id)
        base_node.chunk_id = list(set(base_node.chunk_id))

        merged_results.append(base_node)
    return recursive_batch_merge(merged_results, chunks, chain, safe_chain, preserve_all_data, merge_type)


def pairwise_check_and_merge(
    cluster: List[Node], 
    chunks: List[Document], 
    pairwise_chain, 
    safe_pairwise_chain, 
    preserve_all_data: bool, 
    merge_type: str
) -> List[Node]:

    if len(cluster) <= 1:
        return cluster
    resolved_nodes = [cluster[0]]
    for current_node in cluster[1:]:
        merged = False
        current_chunk = next((c.page_content for c in chunks if c.metadata["chunk_id"] == current_node.chunk_id[0]), "")
        
        for idx, target_node in enumerate(resolved_nodes):
            target_chunk = next((c.page_content for c in chunks if c.metadata["chunk_id"] == target_node.chunk_id[0]), "")
            if merge_type == "names":
                inputs = {
                    "node_a_name": target_node.name, "node_a_context": target_chunk,
                    "node_b_name": current_node.name, "node_b_context": current_chunk
                }
            else:
                inputs = {
                    "node_a_json": MergedNode(name=target_node.name, base_description=target_node.base_description, base_attributes=target_node.base_attributes).model_dump_json(),
                    "node_a_chunk": target_chunk,
                    "node_b_json": MergedNode(name=current_node.name, base_description=current_node.base_description, base_attributes=current_node.base_attributes).model_dump_json(),
                    "node_b_chunk": current_chunk
                }
            if preserve_all_data:
                result = safe_pairwise_chain.invoke(inputs)
            else:
                result = safe_invoke_chain(pairwise_chain, inputs)

            if result and result.name != "":
                merged_node = deepcopy(target_node)
                merged_node.name = result.name
                if merge_type == "full":
                    merged_node.base_description = result.base_description
                    merged_node.base_attributes = result.base_attributes
                
                merged_node.states.extend(current_node.states)
                merged_node.chunk_id.extend(current_node.chunk_id)
                merged_node.chunk_id = list(set(merged_node.chunk_id))

                resolved_nodes[idx] = merged_node
                merged = True
                break 
        if not merged:
            resolved_nodes.append(current_node)
    return resolved_nodes

def merge_similar_nodes(
        chunks: List[Document],
        nodes: List[Node], 
        edges: List[Edge], 
        llm: BaseLanguageModel, 
        embedding_model: Embeddings,
        preserve_all_data: bool = True,
        high_threshold: float = 0.90,
        low_threshold: float = 0.75,
        language: str = "en"
    ) -> Tuple[Dict[str, Node], Dict[str, Edge]]:  
    
    print("Merging nodes.") 
    if not nodes:
        return {}, {}

    if language == "ru":
        chain_cluster = prompt_merging_cluster_en | llm | clean_json | merged_nodes_parser
        safe_chain_cluster = prompt_merging_cluster_en | llm | safe_merged_nodes_parser.parse
        chain_pair = prompt_merging_en | llm | clean_json | merged_nodes_parser
        safe_chain_pair = prompt_merging_en | llm | safe_merged_nodes_parser.parse
    else:
        chain_cluster = prompt_merging_cluster_en | llm | clean_json | merged_nodes_parser
        safe_chain_cluster = prompt_merging_cluster_en | llm | safe_merged_nodes_parser.parse
        chain_pair = prompt_merging_en | llm | clean_json | merged_nodes_parser
        safe_chain_pair = prompt_merging_en | llm | safe_merged_nodes_parser.parse
    id_map = {} 

    high_sim_clusters = cluster_nodes_by_similarity(nodes, embedding_model, high_threshold, use_description=True)
    high_sim_resolved = []
    
    for cluster in high_sim_clusters:
        cluster_ids = [n.id for n in cluster]
        merged_rep = recursive_batch_merge(cluster, chunks, chain_cluster, safe_chain_cluster, preserve_all_data, "full")
        merged_rep.id = cluster[0].id 
        for oid in cluster_ids:
            id_map[oid] = merged_rep.id
        high_sim_resolved.append(merged_rep)

    low_sim_clusters = cluster_nodes_by_similarity(high_sim_resolved, embedding_model, low_threshold, use_description=True)
    final_nodes_dict = {}
    
    for cluster in low_sim_clusters:
        resolved_nodes = pairwise_check_and_merge(cluster, chunks, chain_pair, safe_chain_pair, preserve_all_data, "full")

        for target_node in resolved_nodes:
            final_nodes_dict[target_node.id] = target_node
        
        for original_rep in cluster:
            if original_rep.id not in final_nodes_dict:
                for final_n in resolved_nodes:
                    if set(original_rep.chunk_id).issubset(set(final_n.chunk_id)):
                        for k, v in id_map.items():
                            if v == original_rep.id:
                                id_map[k] = final_n.id
                        break

    merged_edges_dict = {}
    for edge in edges:
        if edge.source in id_map and edge.target in id_map:
            merged_edge = deepcopy(edge)
            merged_edge.source = id_map[edge.source]
            merged_edge.target = id_map[edge.target]

            if merged_edge.source != merged_edge.target:
                merged_edges_dict[merged_edge.id] = merged_edge

    return final_nodes_dict, merged_edges_dict

def extract_entities(
        chunks: List[Document], 
        llm: BaseLanguageModel,
        preserve_all_data: bool = True,
        language: str = "en"
    ) -> Tuple[Dict[str, Node], Dict[str, Edge]]:
    
    all_nodes: Dict[str, Node] = {}
    all_edges: Dict[str, Edge] = {}

    if language == "en":
        chain_entities = prompt_entities_en | llm | clean_json | entities_parser
        safe_chain_entities = prompt_entities_en | llm | safe_entities_parser.parse
    else:
        chain_entities = prompt_entities_ru | llm | clean_json | entities_parser
        safe_chain_entities = prompt_entities_ru | llm | safe_entities_parser.parse

    for idx, chunk in enumerate(chunks):
        
        print(f"[Chunk {idx+1}/{len(chunks)}] Extracting nodes and edges.") #DEBUGGING
        
        coreference_array = resolve_coreference(chunk.page_content, language=language)

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
                continue

        names_to_ids: Dict[str, str] = {}
        for extracted_node in result.nodes:
            node_id = create_id(extracted_node.name)         
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
                names_to_ids[extracted_node.name] = node_id
            else:
                node_id = get_next_unique_node_id(node_id, all_nodes)
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
                names_to_ids[extracted_node.name] = node_id
      
        for extracted_edge in result.edges:
            edge_id_from1to2 = create_id(f"{extracted_edge.node1} {extracted_edge.relation_from1to2} {extracted_edge.node2} edge")
            edge_id_from2to1 = create_id(f"{extracted_edge.node2} {extracted_edge.relation_from2to1} {extracted_edge.node1} edge")
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

def complete_graph(
        chunks: List[Document],
        nodes: Dict[str, Node], 
        edges: Dict[str, Edge],
        llm: BaseLanguageModel,
        preserve_all_data: bool = True,
        language: str = "en"
    ) -> Tuple[Dict[str, Node], Dict[str, Edge]]:

    if language == "en":
        chain_completion = prompt_completing_en | llm | clean_json | completing_graph_parser
        safe_chain_completion = prompt_completing_en | llm | safe_completing_graph_parser.parse
    else:
        chain_completion = prompt_completing_ru | llm | clean_json | completing_graph_parser
        safe_chain_completion = prompt_completing_ru | llm | safe_completing_graph_parser.parse
    
    new_nodes = deepcopy(nodes)
    new_edges = deepcopy(edges)
    names_to_ids: Dict[str, str] = {}

    for node in new_nodes.values():
        names_to_ids[node.name] = node.id

    for idx, chunk in enumerate(chunks):

        print(f"\n[Chunk {idx+1}/{len(chunks)}] Completing nodes and edges.") #DEBUGGING
        chunk_meta_id = chunk.metadata["chunk_id"]
        chunk_nodes = [node for node in nodes.values() if chunk_meta_id in node.chunk_id]
        chunk_edges = [edge for edge in edges.values() if edge.chunk_id == chunk_meta_id]

        entities_for_llm = [{"name": n.name, "description": n.base_description} for n in chunk_nodes]
        relations_for_llm = [{"node1": e.source, "node2": e.target, "relation": e.relation} for e in chunk_edges]

        try:
            if preserve_all_data:
                result: GraphCompletionResult = safe_chain_completion.invoke({
                    "chunk_text": chunk.page_content,
                    "entities_list": entities_for_llm,
                    "existing_relations": relations_for_llm
                })
            else:
                result: GraphCompletionResult = safe_invoke_chain(
                    chain=chain_completion, 
                    inputs={
                        "chunk_text": chunk.page_content,
                        "entities_list": entities_for_llm,
                        "existing_relations": relations_for_llm
                })
                if result == None:
                    continue

            if result and result.missing_entities:
                for missing_entity in result.missing_entities:
                    entity_id = create_id(missing_entity.name)  
                    if entity_id not in new_nodes:
                        new_node = Node(
                            id=entity_id,
                            name=missing_entity.name,
                            type=missing_entity.type,
                            base_description=missing_entity.base_description,
                            base_attributes=missing_entity.base_attributes,
                            states=[],
                            chunk_id=[chunk.metadata["chunk_id"]]
                        )
                        new_nodes[entity_id] = new_node
                        names_to_ids[new_node.name] = entity_id
                    else:
                        entity_id = get_next_unique_node_id(entity_id, new_nodes)
                        new_node = Node(
                            id=entity_id,
                            name=missing_entity.name,
                            type=missing_entity.type,
                            base_description=missing_entity.base_description,
                            base_attributes=missing_entity.base_attributes,
                            states=[],
                            chunk_id=[chunk.metadata["chunk_id"]]
                        )
                        new_nodes[entity_id] = new_node
                        names_to_ids[new_node.name] = entity_id
            
            if result and result.missing_relations:
                for missing in result.missing_relations:
                    edge_id_1to2 = create_id(f"{missing.node1} {missing.relation_from1to2} {missing.node2} edge")
                    edge_id_2to1 = create_id(f"{missing.node2} {missing.relation_from2to1} {missing.node1} edge")

                    if edge_id_1to2 in new_edges:
                        continue

                    node1_id = names_to_ids.get(missing.node1)
                    node2_id = names_to_ids.get(missing.node2)
                    if not node1_id or not node2_id:
                        continue

                    new_edges[edge_id_1to2] = Edge(
                        id=edge_id_1to2,
                        source=node1_id,
                        target=node2_id,
                        relation=missing.relation_from1to2,
                        description=missing.description,
                        weight=missing.weight,
                        chunk_id=chunk.metadata["chunk_id"]
                    )
                    new_edges[edge_id_2to1] = Edge(
                        id=edge_id_2to1,
                        source=node2_id,
                        target=node1_id,
                        relation=missing.relation_from2to1,
                        description=missing.description,
                        weight=missing.weight,
                        chunk_id=chunk.metadata["chunk_id"]
                    )
            
        except Exception as e:
            continue

    return new_nodes, new_edges

def apply_event_impact_on_graph(
        graph: KnowledgeGraph, 
        impact: EventImpact, 
        event: Node
    ) -> None:

    print(f"Applying events impacts: {impact.event_name}") #DEBUGGING
    
    event_id = event.id
    event_name = event.name
    
    if impact.affected_nodes:
        for affected_node in impact.affected_nodes:
            existing_node = graph.get_node_by_id(affected_node.id)
            if existing_node:
                new_state = State(
                    sid=f"{event_id}_{affected_node.id}_{len(existing_node.states)}",
                    current_description=affected_node.new_current_description,
                    current_attributes=affected_node.new_current_attributes,
                    time_start_event=affected_node.time_start_event if affected_node.time_start_event else event_name,
                    time_end_event=affected_node.time_end_event
                )
                
                if new_state.time_start_event and not new_state.time_end_event:
                    for prev_state in existing_node.states:
                        if prev_state.time_end_event is None:
                            prev_state.time_end_event = event_name
                            break
                
                graph.update_node_states(affected_node.id, new_state)
    
    if impact.affected_edges:
        for affected_edge in impact.affected_edges:
            existing_edge = graph.get_edge_by_id(affected_edge.id)
            if existing_edge:
                new_description = affected_edge.new_description if affected_edge.new_description else existing_edge.description
                
                time_start = affected_edge.time_start_event if affected_edge.time_start_event else existing_edge.time_start_event
                time_end = affected_edge.time_end_event if affected_edge.time_end_event else existing_edge.time_end_event
                
                graph.update_edge_times(
                    affected_edge.id, 
                    new_description, 
                    time_start_event=time_start,
                    time_end_event=time_end
                )

def extract_events_impact(
        chunks: List[Document],
        nodes: List[Node],
        edges: List[Edge], 
        llm: BaseLanguageModel,
        preserve_all_data: bool = True,
        language: str = "en"
    ) -> List[EventImpact]:

    event_impacts_all = []
    
    if language == "en":
        chain_event = prompt_events_en | llm | clean_json | events_parser
        safe_chain_event = prompt_events_en | llm | safe_events_parser.parse
    else:
        chain_event = prompt_events_ru | llm | clean_json | events_parser
        safe_chain_event = prompt_events_ru | llm | safe_events_parser.parse

    for idx, chunk in enumerate(chunks):
        print(f"[Chunk {idx+1}/{len(chunks)}] Extracting events impact.")

        chunk_meta_id = chunk.metadata["chunk_id"]
        chunk_nodes = [node for node in nodes if chunk_meta_id in node.chunk_id]
        chunk_edges = [edge for edge in edges if edge.chunk_id == chunk_meta_id]
        
        event_names = [node.name for node in chunk_nodes if node.type == "event"]
        entities_nodes = [node for node in chunk_nodes if node.type != "event"]
        
        entities_input_nodes = [create_input_node(node) for node in entities_nodes]
        chunk_input_edges = [create_input_edge(edge) for edge in chunk_edges]
        
        if len(event_names) > 0:
            if preserve_all_data:
                events_impacts: EventsSubgraph = safe_chain_event.invoke({
                    "chunk_text": chunk.page_content,
                    "events_list": event_names,
                    "entities_list": entities_input_nodes,
                    "edges_list": chunk_input_edges
                })
            else:
                events_impacts = safe_invoke_chain(
                    chain=chain_event, 
                    inputs={
                        "chunk_text": chunk.page_content,
                        "events_list": event_names,
                        "entities_list": entities_input_nodes,
                        "edges_list": chunk_input_edges
                    }
                )
            if events_impacts and len(events_impacts.events_with_impact) > 0:
                for impact in events_impacts.events_with_impact:
                    event_impacts_all.append(impact)
    return event_impacts_all

def extract_graph(
        chunks: List[Document],       
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        graph_class = NetworkXGraph,
        preserve_all_data: bool = True,
        language: str = "en"
    ) -> KnowledgeGraph:
    
    graph = graph_class()
    nodes, edges = extract_entities(
        chunks=chunks, 
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )

    completed_nodes, completed_edges = complete_graph(
        chunks=chunks,
        nodes=nodes,
        edges=edges,
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )

    all_nodes = [n for n in completed_nodes.values()]
    all_edges = [e for e in completed_edges.values()]
    
    merged_result = merge_similar_nodes(
        chunks=chunks,
        nodes=all_nodes, 
        edges=all_edges, 
        llm=llm, 
        embedding_model=embedding_model,
        preserve_all_data=preserve_all_data,
        language=language
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
        preserve_all_data=preserve_all_data,
        language=language
    )  

    events_only = [node for node in nodes_in_graph if node.type == "event"]
    for event in events_impacts:     
        for event_in_graph in events_only:
            print (event_in_graph.name, event.event_name)
            if event_in_graph.name == event.event_name:
                apply_event_impact_on_graph(graph, event, event_in_graph)

    return graph

def extract_graph_from_nodes(
        chunks: List[Document],       
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        graph_class = NetworkXGraph,
        preserve_all_data: bool = True,
        language: str = "en"
    ) -> KnowledgeGraph:
    
    graph = graph_class()

    nodes_names = extract_entities_names(
        chunks=chunks, 
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )
    print("---NODES---")
    for node in nodes_names.values():
        print(node)

    merged_nodes_names = merge_similar_entities_names(
        chunks=chunks,
        nodes=nodes_names.values(), 
        llm=llm,
        embedding_model=embedding_model,
        preserve_all_data=preserve_all_data,
        language=language
    )
    print("---MERGED NODES---")
    print(merged_nodes_names)

    nodes, edges = extract_graph_info(
        chunks=chunks,
        nodes=merged_nodes_names,
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )

    completed_nodes, completed_edges = complete_graph(
        chunks=chunks,
        nodes=nodes,
        edges=edges,
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )

    all_nodes = [n for n in completed_nodes.values()]
    all_edges = [e for e in completed_edges.values()]

    merged_result = merge_similar_nodes(
        chunks=chunks,
        nodes=all_nodes, 
        edges=all_edges, 
        llm=llm, 
        embedding_model=embedding_model,
        preserve_all_data=preserve_all_data,
        language=language
    )
    for n in merged_result[0].values():
        graph.add_node(n)
    for e in merged_result[1].values():
        if e.source and e.target:
            graph.add_edge(e)

    print(f"Graph built with {len(all_nodes)} nodes and {len(all_edges)} edges.") #DEBUGGING
    
    nodes_in_graph = graph.get_all_nodes()
    edges_in_graph = graph.get_all_edges()
    events_impacts = extract_events_impact(
        chunks=chunks, 
        nodes=nodes_in_graph, 
        edges=edges_in_graph, 
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )  

    events_only = [node for node in nodes_in_graph if node.type == "event"]
    for event in events_impacts:
        for event_in_graph in events_only:
            print (event_in_graph.name, event.event_name)
            if event_in_graph.name == event.event_name:
                apply_event_impact_on_graph(graph, event, event_in_graph)

    return graph

def update_graph(
        chunks: List[Document],       
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        graph: KnowledgeGraph,
        preserve_all_data: bool = True,
        language: str = "en"
    ) -> None:
    
    nodes, edges = extract_entities(
        chunks=chunks, 
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )

    completed_nodes, completed_edges = complete_graph(
        chunks=chunks,
        nodes=nodes,
        edges=edges,
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )
    
    all_nodes = graph.get_all_nodes()
    for n in completed_nodes.values():
        all_nodes.append(n)
    
    all_edges = graph.get_all_edges()
    for e in completed_edges.values():
        all_edges.append(e)

    merged_result = merge_similar_nodes(
        chunks=chunks,
        nodes=all_nodes, 
        edges=all_edges, 
        llm=llm, 
        embedding_model=embedding_model,
        preserve_all_data=preserve_all_data,
        language=language
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
        preserve_all_data=preserve_all_data,
        language=language
    )  
    
    events_only = [node for node in nodes_in_graph if node.type == "event"]
    for event in events_impacts:
        for event_in_graph in events_only:
            if event_in_graph.name == event.event_name:
                apply_event_impact_on_graph(graph, event, event_in_graph)


def update_graph_from_nodes(
        chunks: List[Document],       
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        graph: KnowledgeGraph,
        preserve_all_data: bool = True,
        language: str = "en"
    ) -> None:
    
    nodes_names = extract_entities_names(
        chunks=chunks, 
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )

    merged_nodes_names = merge_similar_entities_names(
        chunks=chunks,
        nodes=nodes_names.values(), 
        llm=llm,
        embedding_model=embedding_model,
        preserve_all_data=preserve_all_data,
        language=language
    )

    nodes, edges = extract_graph_info(
        chunks=chunks,
        nodes=merged_nodes_names,
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )

    completed_nodes, completed_edges = complete_graph(
        chunks=chunks,
        nodes=nodes,
        edges=edges,
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )
    
    all_nodes = graph.get_all_nodes()
    for n in completed_nodes.values():
        all_nodes.append(n)
    
    all_edges = graph.get_all_edges()
    for e in completed_edges.values():
        all_edges.append(e)

    merged_result = merge_similar_nodes(
        chunks=chunks,
        nodes=all_nodes, 
        edges=all_edges, 
        llm=llm, 
        embedding_model=embedding_model,
        preserve_all_data=preserve_all_data,
        language=language
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
        preserve_all_data=preserve_all_data,
        language=language
    )  
    
    events_only = [node for node in nodes_in_graph if node.type == "event"]
    for event in events_impacts:
        for event_in_graph in events_only:
            if event_in_graph.name == event.event_name:
                apply_event_impact_on_graph(graph, event, event_in_graph)


def merge_several_nodes_in_graph(
        graph: KnowledgeGraph,
        nodes_to_merge: List[Node],
        llm: BaseLanguageModel, 
        embedding_model: Embeddings,
        preserve_all_data: bool = True,
        language: str = "en"
    ) -> None:  
    
    if language == "en":
        chain_merging = prompt_merging_in_graph_en | llm | clean_json | merged_in_graph_nodes_parser
        safe_chain_merging = prompt_merging_in_graph_en | llm | safe_merged_in_graph_nodes_parser.parse
    else:
        chain_merging = prompt_merging_ru | llm | clean_json | merged_nodes_parser
        safe_chain_merging = prompt_merging_ru | llm | safe_merged_nodes_parser.parse

    if not nodes_to_merge:
        return

    node_ids = [node.id for node in nodes_to_merge]
    result_node = nodes_to_merge[0]
    for i in range(1, len(nodes_to_merge)):
        node_i = nodes_to_merge[i]
        node_i_for_llm = MergedInGraphNode(
            name=node_i.name,
            type=node_i.type,
            base_description=node_i.base_description,
            base_attributes=node_i.base_attributes,
            states=node_i.states
        )
        node_j_for_llm = MergedInGraphNode(
            name=result_node.name,
            type=result_node.type,
            base_description=result_node.base_description,
            base_attributes=result_node.base_attributes,
            states=result_node.states
        )

        if preserve_all_data:
            merged_node: MergedInGraphNode = safe_chain_merging.invoke({
                "node_a_json": node_i_for_llm.model_dump_json(),
                "node_b_json": node_j_for_llm.model_dump_json(),
            })
        else:
            merged_node: MergedInGraphNode = safe_invoke_chain(
                chain=chain_merging, 
                inputs={
                    "node_a_json": node_i_for_llm.model_dump_json(),
                    "node_b_json": node_j_for_llm.model_dump_json(),
            })
        result_node = Node(
            id=create_id(merged_node.name),
            name=merged_node.name,
            type=merged_node.type,
            base_description=merged_node.base_description,
            base_attributes=merged_node.base_attributes,
            states=merged_node.states,
            chunk_id=list(set(result_node.chunk_id + node_i.chunk_id))
        )
    
    edges_to_rewire = []
    for u, v, key, attrs in graph.graph.edges(data=True, keys=True):
        if u in node_ids or v in node_ids:
            new_source = result_node.id if u in node_ids else u
            new_target = result_node.id if v in node_ids else v
            new_attrs = deepcopy(attrs)
            if "data" in new_attrs:
                new_attrs["data"]["source"] = new_source
                new_attrs["data"]["target"] = new_target
            edges_to_rewire.append((new_source, new_target, key, new_attrs))
    for node_id in node_ids:
        graph.remove_node(node_id)
    for new_source, new_target, key, attrs in edges_to_rewire:
        graph.graph.add_edge(new_source, new_target, key=key, **attrs)
    graph.add_node(result_node)

    update_embeddings(graph, graph.get_vector_db(), embedding_model=embedding_model)
    return



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
        text = f"{node.name} {node.base_description}"
        all_ids.append(f"full_{node.id}")
        all_documents.append(text)
        all_metadatas.append({
            "type": "node",
            "node_type": node.type,
            "name": node.name,
            "original_id": node.id
        })
        text = f"{node.name}"
        all_ids.append(f"name_{node.id}")
        all_documents.append(text)
        all_metadatas.append({
            "type": "node",
            "node_type": node.type,
            "name": node.name,
            "original_id": node.id
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
    print(all_ids)
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
        graph_ids.append(f"full_{node.id}")
        graph_docs.append(f"{node.name} {node.base_description}")
        graph_metadatas.append({
            "type": "node",
            "node_type": node.type,
            "name": node.name
        })
        graph_ids.append(f"name_{node.id}")
        graph_docs.append(f"{node.name}")
        graph_metadatas.append({
            "type": "node",
            "node_type": node.type,
            "name": node.name
        })
    for edge in graph.get_all_edges():
        graph_ids.append(edge.id)
        graph_docs.append(f"{edge.source} {edge.relation} {edge.target}")
        graph_metadatas.append({
            "type": "edge",
            "relation": edge.relation,
            "source": edge.source,
            "target": edge.target
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