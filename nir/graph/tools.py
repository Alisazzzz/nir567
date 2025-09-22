#All stuff about working with graphs is here

from typing import List
from langchain.schema import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.language_models.base import BaseLanguageModel

def extract_graph(chunks: List[Document], llm: BaseLanguageModel) -> List[GraphDocument]:
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