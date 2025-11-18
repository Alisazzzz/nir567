#All stuff with loading documents is here

from typing import List

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.telegram import text_to_docs

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


#Loaders
def loadCSV(path: str, encoding: str = "utf-8") -> List[Document]:
    loader = CSVLoader(file_path=path, encoding=encoding)
    data = loader.load()
    return data

def loadTXT(path: str, encoding: str = "utf-8") -> List[Document]:
    loader = TextLoader(file_path=path, encoding=encoding)
    data = loader.load()
    return data

def convertFromString(string: str, encoding: str = "utf-8") -> List[Document]:
    data = text_to_docs(string)
    return data

#Chunk splitter
def to_chunk(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks

def to_chunk_unique_id(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 50, start_chunk_id: int = 0) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    current_id = start_chunk_id
    new_chunks = []
    for chunk in chunks:
        new_metadata = dict(chunk.metadata) if chunk.metadata else {}
        new_metadata["chunk_id"] = current_id
        new_chunks.append(Document(page_content=chunk.page_content, metadata=new_metadata))
        current_id += 1
    return new_chunks