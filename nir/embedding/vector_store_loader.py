#All stuff for creating every vector store is here

from typing import Dict
from pydantic import BaseModel, Field

from nir.embedding.vector_storages.chroma_db import ChromaVectorStore
from nir.embedding.vector_store import VectorStore


class VectorStoreInfo(BaseModel):
    type: str
    info: Dict[str, str] = Field(default_factory=dict)

def create_vector_store(config: VectorStoreInfo) -> VectorStore:
    vector_store = None
    match config.type:
        case "chromadb":  
            vector_store = create_chromadb_store(config.info)
        case _:
            raise ValueError(f"Неизвестный тип эмбеддингов: {type}")
    return vector_store
        
def create_chromadb_store(info: Dict[str, str]) -> ChromaVectorStore:
    return ChromaVectorStore(collection_name=info.get("name", ""), persist_directory=info.get("path", ""))