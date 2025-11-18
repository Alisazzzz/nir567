# Abstract class for vector store

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    @abstractmethod
    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> None:
        pass

    @abstractmethod
    def delete_embeddings(self, ids: List[str]) -> None:
        pass

    @abstractmethod
    def update_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> None:
        pass

    @abstractmethod
    def get_all_ids(self) -> List[str]:
        pass

    @abstractmethod
    def get_metadata(self, id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def persist(self) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        pass