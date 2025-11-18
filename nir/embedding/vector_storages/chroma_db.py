import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from nir.embedding.vector_store import VectorStore

class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str = "graph_embeddings",
        persist_directory: str = "assets/databases/chroma_db"
    ):
        self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
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

    def delete_embeddings(self, ids: List[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def update_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> None:
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def get_all_ids(self) -> List[str]:
        results = self.collection.get()  # returns all items
        return results.get("ids", [])

    def get_metadata(self, id: str) -> Dict[str, Any]:
        results = self.collection.get(ids=[id])
        if results and "metadatas" in results and results["metadatas"]:
            return results["metadatas"][0]
        return {}

    def persist(self) -> None:
        pass

    def get_collection(self):
        return self.collection
    
    def search(
            self,
            query_embedding: List[float],
            top_k: int = 5
        ) -> List[Dict[str, Any]]:
        
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i],
            })
        return hits
