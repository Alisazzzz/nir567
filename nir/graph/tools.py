#All stuff about working with graphs is here

from typing import List
from langchain.schema import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.language_models.base import BaseLanguageModel

def OLD_extract_graph(chunks: List[Document], llm: BaseLanguageModel) -> List[GraphDocument]:
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(chunks)
    return graph_documents

def update_graph(new_graph: List[GraphDocument]):
    return

def remove_subgraph(subgraph: List[GraphDocument]):
    return

def update_states(event: str):
    return

#Some functions i'm not sure about
def add_node(node):
    return
def add_edge(edge):
    return
def remove_nodes_by_names(names: List[str]):
    return
def remove_nodes_by_ids(ids: List[int]):
    return
def remove_edges_by_id(ids: List[int]):
    return

from typing import List, Dict
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document

from nir.graph.graph_structures import NodeType, Node, Edge
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.graph.graph_storages.networkx_graph import NetworkXGraph

def normalize_id(text: str) -> str:
    return text.lower().replace(" ", "_")

class GraphExtractionResult(BaseModel):
    nodes: List[Node] = Field(default_factory=list, description="List of nodes")
    edges: List[Edge] = Field(default_factory=list, description="List of edges")

parser = PydanticOutputParser(pydantic_object=GraphExtractionResult)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
        "You are an expert in knowledge extraction. Your task is to extract entities and relationships from the given text.\n\n"
        
        "ENTITY TYPES (use EXACTLY these values for 'type'):\n"
        "- character\n"
        "- group\n"
        "- location\n"
        "- environment_element\n"
        "- item\n\n"
        
        "NEVER extract events as nodes. Only persistent entities.\n\n"
        
        "OUTPUT FORMAT REQUIREMENTS:\n"
        "- Every node MUST have: id, name, type, description (can be empty string \"\"), attributes (empty dict {{}} if none), states (empty list [] if none).\n"
        "- Every edge MUST have: id, source, target, relation, description (can be \"\"), weight (default 1.0), time_start_event (null if unknown), time_end_event (null if unknown).\n"
        "- Use normalized lowercase IDs with underscores (e.g., 'alice_cooper', 'dark_forest').\n"
        "- The 'source' and 'target' in edges must match node 'id' values exactly.\n\n"
        
        "EXAMPLE (for reference only — do not copy this data):\n"
        "Input text: \"Alice Cooper lives in a dark forest. She owns a magic lamp.\"\n"
        "Output:\n"
        "{{\n"
        "  \"nodes\": [\n"
        "    {{\n"
        "      \"id\": \"alice_cooper\",\n"
        "      \"name\": \"Alice Cooper\",\n"
        "      \"type\": \"character\",\n"
        "      \"description\": \"\",\n"
        "      \"attributes\": {{}},\n"
        "      \"states\": []\n"
        "    }},\n"
        "    {{\n"
        "      \"id\": \"dark_forest\",\n"
        "      \"name\": \"Dark Forest\",\n"
        "      \"type\": \"location\",\n"
        "      \"description\": \"\",\n"
        "      \"attributes\": {{}},\n"
        "      \"states\": []\n"
        "    }},\n"
        "    {{\n"
        "      \"id\": \"magic_lamp\",\n"
        "      \"name\": \"Magic Lamp\",\n"
        "      \"type\": \"item\",\n"
        "      \"description\": \"\",\n"
        "      \"attributes\": {{}},\n"
        "      \"states\": []\n"
        "    }}\n"
        "  ],\n"
        "  \"edges\": [\n"
        "    {{\n"
        "      \"id\": \"alice_cooper_lives_in_dark_forest\",\n"
        "      \"source\": \"alice_cooper\",\n"
        "      \"target\": \"dark_forest\",\n"
        "      \"relation\": \"lives_in\",\n"
        "      \"description\": \"\",\n"
        "      \"weight\": 1.0,\n"
        "      \"time_start_event\": null,\n"
        "      \"time_end_event\": null\n"
        "    }},\n"
        "    {{\n"
        "      \"id\": \"alice_cooper_owns_magic_lamp\",\n"
        "      \"source\": \"alice_cooper\",\n"
        "      \"target\": \"magic_lamp\",\n"
        "      \"relation\": \"owns\",\n"
        "      \"description\": \"\",\n"
        "      \"weight\": 1.0,\n"
        "      \"time_start_event\": null,\n"
        "      \"time_end_event\": null\n"
        "    }}\n"
        "  ]\n"
        "}}\n\n"
        
        "STRICTLY follow this structure. Output ONLY valid JSON. Do not add explanations."
    ),
    ("human", "Analyze the following text and extract nodes and edges for a knowledge graph.\n\nText:\n{chunk}\n\n{format_instructions}")
]).partial(format_instructions=parser.get_format_instructions())

def extract_graph(chunks: List[Document], llm: BaseLanguageModel, graph_class=NetworkXGraph) -> KnowledgeGraph:  
    graph = graph_class()
    chain = prompt | llm | parser
    all_nodes: Dict[str, Node] = {}
    all_edges: Dict[str, Edge] = {}
    max_stages = len(chunks)
    current_stage = -1

    for chunk in chunks:
        current_stage += 1
        try:
            result: GraphExtractionResult = chain.invoke({"chunk": chunk.page_content})
            print(str(current_stage) + " | " + str(max_stages))

            for node in result.nodes:
                norm_id = normalize_id(node.id)
                if norm_id not in all_nodes:
                    all_nodes[norm_id] = node
                else:
                    pass
            for edge in result.edges:
                norm_id = normalize_id(edge.id)
                norm_source = normalize_id(edge.source)
                norm_target = normalize_id(edge.target)
                if norm_id not in all_edges:
                    all_edges[norm_id] = edge
                if norm_source not in all_nodes:
                    all_nodes[norm_source] = Node(
                        id=norm_source,
                        name=norm_source,
                        type=NodeType.item,
                        description="Automatically created node",
                        attributes= {},
                        states = []
                    )
                if norm_target not in all_nodes:
                    all_nodes[norm_target] = Node(
                        id=norm_target,
                        name=norm_target,
                        type=NodeType.item,
                        description="Automatically created node",
                        attributes= {},
                        states = []
                    )                  
        except Exception as e:
            print(f"Error with chunk: {e}")
            continue

    for node in all_nodes.values():
        graph.add_node(node)
    for edge in all_edges.values():
       graph.add_edge(edge)
    
    return graph


from typing import List, Dict, Any
from langchain_core.embeddings import Embeddings
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.embedding.vector_store import VectorStore
from nir.graph.graph_structures import Node, Edge

def create_embeddings_from_graph(graph: KnowledgeGraph, vector_store: VectorStore, embedding_model: Embeddings) -> None:
    nodes = graph.get_all_nodes()
    edges = graph.get_all_edges()

    all_ids: List[str] = []
    all_documents: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []

    for node in nodes:
        text = f"Entity: {node.name}. Description: {node.description or 'No description'}"
        all_ids.append(node.id)
        all_documents.append(text)
        all_metadatas.append({
            "type": "node",
            "node_type": node.type,
            "name": node.name
        })

    for edge in edges:
        text = (
            f"Relationship: {edge.source} --[{edge.relation}]--> {edge.target}. "
            f"Description: {edge.description or 'No description'}"
        )
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

    return