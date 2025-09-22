#All stuff with loading documents is here

from typing import List

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader

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


#Chunk splitter
def to_chunk(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks