#All stuff with graph extraction is here

import numpy as np
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from sentence_transformers import util
import spacy
from rapidfuzz import fuzz
import nltk
from nltk.corpus import wordnet

from nir.graph.graph_structures import NodeType, Node, Edge, EventImpact, State, EventsSubgraph
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.graph.graph_storages.networkx_graph import NetworkXGraph

from nir.prompts import extraction_prompts

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

def normalize_id(text: str) -> str:
    return text.lower().replace(" ", "_").strip()

def cosine_sim(text1: str, text2: str, model) -> float:
    emb1 = np.array(model.embed_query(text1))
    emb2 = np.array(model.embed_query(text2))
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


class GraphExtractionResult(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

entities_parser = PydanticOutputParser(pydantic_object=GraphExtractionResult)

prompt_hybrid = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_HYBRID),
    ("human",
        "Text:\n{chunk}\n\n"
        "Entities (with possible synonyms):\n{entities_json}\n\n"
        "{format_instructions}")
]).partial(format_instructions=entities_parser.get_format_instructions())

prompt_simple = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_SIMPLE),
    ("human", "Text:\n{chunk}\n\n{format_instructions}")
]).partial(format_instructions=entities_parser.get_format_instructions())


events_parser = PydanticOutputParser(pydantic_object=EventsSubgraph)

prompt_events = ChatPromptTemplate.from_messages([
    ("system", extraction_prompts.SYSTEM_PROMPT_EVENTS),
    ("human",
        "Chunk ID: {chunk_id}\n\n"
        "Text:\n{chunk}\n\n"
        "Entities (from this chunk):\n{entities_json}\n\n"
        "Edges (between these entities):\n{edges_json}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=events_parser.get_format_instructions())


class GraphExtractor:
    def __init__(self,
                 llm: BaseLanguageModel,
                 embedder: Embeddings,
                 graph_class=NetworkXGraph,
                 similarity_threshold: float = 0.85,
                 mode: str = "hybrid"):
        #:param mode: 'hybrid' uses spaCy + coref before LLM;
        #             'llm_only' directly extracts with LLM
        
        self.llm = llm
        self.graph_class = graph_class
        self.similarity_threshold = similarity_threshold
        self.mode = mode
        self.embedder = embedder
        self.spacy_nlp = spacy.load("en_core_web_sm") if mode == "hybrid" else None
        self.chain_hybrid = prompt_hybrid | llm | entities_parser
        self.chain_simple = prompt_simple | llm | entities_parser
        self.chain_event = prompt_events | llm | events_parser


    def _are_synonyms(self, a: str, b: str, threshold: float = 0.85) -> bool:
        if not a or not b:
            return False
        a, b = a.strip().lower(), b.strip().lower()
        if a == b:
            return True

        if fuzz.token_sort_ratio(a, b) >= 90:
            return True

        if len(a.split()) == 1 and len(b.split()) == 1:
            synsets_a = wordnet.synsets(a)
            synsets_b = wordnet.synsets(b)
            if synsets_a and synsets_b:
                for sa in synsets_a:
                    for sb in synsets_b:
                        if sa == sb:
                            return True

        try:
            emb_a = np.array(self.embedder.embed_query(a))
            emb_b = np.array(self.embedder.embed_query(b))
            sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
            return sim >= threshold
        except Exception:
            return False


    def _extract_entities_and_synonyms(self, text: str, chunk_id: str) -> List[Node]:

        doc = self.spacy_nlp(text)
        entities: Dict[str, Node] = {}

        for ent in doc.ents:
            node_id = normalize_id(ent.text)
            if node_id not in entities:
                entities[node_id] = Node(
                    id=node_id,
                    name=ent.text,
                    type=NodeType.item,
                    description="",
                    synonyms=[],
                    attributes={},
                    states=[],
                    chunk_id=chunk_id,
                )
        return list(entities.values())



    def _merge_similar_nodes(self, nodes: List[Node]) -> List[Node]:
    
        if not nodes:
            return []
        merged = []
        used = set()

        for i, node_i in enumerate(nodes):
            if i in used:
                continue
            cluster = [node_i]
            used.add(i)
            for j, node_j in enumerate(nodes):
                if j in used:
                    continue
                if self._are_synonyms(node_i.name, node_j.name, threshold=self.similarity_threshold):
                    cluster.append(node_j)
                    used.add(j)

            if len(cluster) > 1:
                merged_name = "/".join(sorted(set(n.name for n in cluster)))
                merged_synonyms = list(set(sum([n.synonyms for n in cluster], []))) + [n.name for n in cluster if n.name != merged_name]
                merged_node = Node(
                    id=normalize_id(merged_name),
                    name=merged_name,
                    type=NodeType.item,
                    description="Merged from: " + ", ".join(n.name for n in cluster),
                    synonyms=merged_synonyms,
                    attributes={},
                    states=[],
                    chunk_id=node_i.chunk_id,
                )
                merged.append(merged_node)
            else:
                merged.append(cluster[0])

        return merged



    def _apply_event_impacts(self, graph: KnowledgeGraph, events: List[EventImpact]) -> None:

        for idx, event in enumerate(events):
            eid = event.event_id

            for node in event.affected_nodes:
                try:
                    node = graph.get_node_by_id(node.id)
                except Exception:
                    continue

                new_description = node.description

                new_state = State(
                    sid=f"{eid}_{node.id}",
                    attributes=node.attributes,
                    time_start=eid,
                    time_end=None
                )
                graph.update_node_state(node.id, new_description, new_state)

            for edge in event.affected_edges:
                try:
                    node = graph.get_edge_by_id(node.id)
                except Exception:
                    continue

                new_description = edge.description
                
                if (edge.time_start_event and edge.time_end_event):
                    graph.update_edge_times(edge.id, time_start_event=edge.time_start_event, time_end_event=edge.time_end_event)
                elif (edge.time_start_event):
                    graph.update_edge_times(edge.id, time_start_event=edge.time_start_event)
                elif (edge.time_end_event):
                    graph.update_edge_times(edge.id, time_end_event=edge.time_end_event)



    def _extract_entities(self, chunks: List[Document], graph: KnowledgeGraph) -> None:

        all_nodes: Dict[str, Node] = {}
        all_edges: Dict[str, Edge] = {}

        for idx, chunk in enumerate(chunks):
            print(f"[Chunk {idx+1}/{len(chunks)}] Processing...")

            if self.mode == "hybrid":
                entities = self._extract_entities_and_synonyms(chunk.page_content, str(idx))
                #entities = self._merge_similar_nodes(entities)
                entities_json = [e.model_dump() for e in entities]

                result: GraphExtractionResult = self.chain_hybrid.invoke({
                    "chunk": chunk.page_content,
                    "entities_json": entities_json
                })
            else:
                result: GraphExtractionResult = self.chain_simple.invoke({"chunk": chunk.page_content})

            for node in result.nodes:
                norm_id = normalize_id(node.id)
                if norm_id not in all_nodes:
                    all_nodes[norm_id] = Node(**node.model_dump())
                    all_nodes[norm_id].chunk_id = str(idx)
            
            for edge in result.edges:
                norm_id = normalize_id(edge.id)
                if norm_id not in all_edges:
                    all_edges[norm_id] = edge
                    all_edges[norm_id].chunk_id = str(idx)

        for n in all_nodes.values():
            graph.add_node(n)
        for e in all_edges.values():
            graph.add_edge(e)

        print(f"Graph built with {len(all_nodes)} nodes and {len(all_edges)} edges.")



    def _extract_events(self, chunks: List[Document], graph: KnowledgeGraph) -> None:

        for idx, chunk in enumerate(chunks):
            chunk_id = idx
            print(f"[Chunk {idx+1}/{len(chunks)}] Extracting events...")

            try:
                chunk_nodes = [
                    node for node in graph.get_all_nodes()
                    if getattr(node, "chunk_id", None) == chunk_id
                ]
                chunk_edges = [
                    edge for edge in graph.get_all_edges()
                    if getattr(edge, "chunk_id", None) == chunk_id
                ]
            except Exception as e:
                print(f"Error collecting nodes/edges for chunk {chunk_id}: {e}")
                continue

            entities_json = [n.model_dump() for n in chunk_nodes]
            edges_json = [e.model_dump() for e in chunk_edges]

            try:
                event_subgraph: EventsSubgraph = self.chain_event.invoke({
                    "chunk": chunk.page_content,
                    "chunk_id": chunk_id,
                    "entities_json": entities_json,
                    "edges_json": edges_json
                })
            except Exception as e:
                print(f"LLM extraction error for chunk {chunk_id}: {e}")
                continue

            for node in event_subgraph.nodes:
                node.chunk_id = chunk_id
                if not any(n.id == node.id for n in graph.get_all_nodes()):
                    graph.add_node(node)

            for edge in event_subgraph.edges:
                edge.chunk_id = chunk_id
                if not any(e.id == edge.id for e in graph.get_all_edges()):
                    graph.add_edge(edge)        

            try:
                self._apply_event_impacts(graph, event_subgraph.events_with_impact)
            except Exception as e:
                print(f"Error applying impacts for chunk {chunk_id}: {e}")
                continue

        print(f"Events integrated for chunk {chunk_id}: "
              f"{len(event_subgraph.nodes)} event nodes, "
              f"{len(event_subgraph.edges)} event edges.")
        
    print("Event extraction and integration complete.")



    def extract_graph(self, chunks: List[Document]) -> KnowledgeGraph:

        graph = self.graph_class()

        self._extract_entities(chunks, graph)
        self._extract_events(chunks, graph)

        return graph
