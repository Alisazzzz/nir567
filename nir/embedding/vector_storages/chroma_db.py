#In case of using ChromaDB as vectore storage

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from nir.embedding.vector_store import VectorStore

class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name: str = "graph_embeddings", persist_directory: str = "assets/databases/chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.persist_directory = persist_directory

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> None:
        if not (len(ids) == len(embeddings) == len(metadatas) == len(documents)):
            raise ValueError("All input lists must have the same length")
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def persist(self) -> None:
        pass

    def get_collection(self):
        return self.collection