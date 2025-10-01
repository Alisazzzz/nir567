#Abstract class for knowledge graph

from typing import Iterable, List
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_chroma import Chroma
import chromadb

from nir.graph.graph_structures import Node, Edge, State
from abc import ABC, abstractmethod

class KnowledgeGraph(ABC):
    @abstractmethod
    def add_node(self, node: Node) -> None:
        pass

    @abstractmethod
    def add_edge(self, edge: Edge) -> None:
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Node:
        pass

    @abstractmethod
    def get_edges(self, source: str, target: str) -> List[Edge]:
        pass

    @abstractmethod
    def update_node_state(self, node_id: str, new_state: State) -> None:
        pass
    
    @abstractmethod
    def remove_node(self, node_id: str) -> None:
        pass

    @abstractmethod
    def remove_edge(self, ebde_id: str) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass




def create_embeddings(texts: Iterable[str], embeddings) -> VectorStoreRetriever:
    client = chromadb.PersistentClient(path="./chroma_langchain_db")

    vector_store = Chroma(
        collection_name="test_collection",
        embedding_function=embeddings,
        persist_directory="../../assets/databases/chroma_langchain_db",
    )

    vector_store.add_texts(
        texts=texts,
        ids=[f"id_{i}" for i in range(len(texts))],
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever