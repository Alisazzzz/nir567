#I don't know may be I'm a bit confused by this structure

from typing import Iterable
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_chroma import Chroma
import chromadb

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