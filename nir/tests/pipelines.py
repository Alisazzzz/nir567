from abc import ABC, abstractmethod
from typing import Optional, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from nir.core.answers_generator import generate_answer_based_on_plan, generate_plan
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.core.context_retriever import form_context_with_llm, form_context_without_llm


class BasePipeline(ABC):
    
    @abstractmethod
    def generate(self, query: str) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class BasicLLMPipeline(BasePipeline):

    def __init__(self, llm: BaseLanguageModel, context: str, name: str = "Basic LLM"):
        self.llm = llm
        self.context = context
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(self, query: str) -> str:
        prompt = f"{self.context}\n\n Question: {query}"
        response = self.llm.invoke(prompt)
        return str(response)


class StandardRAGPipeline(BasePipeline):
    def __init__(
        self,
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        filepath: str,
        name: str = "Standard RAG",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self._name = name

        loader = TextLoader(filepath, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks: List[Document] = text_splitter.split_documents(documents)

        self.vectorstore = FAISS.from_documents(chunks, embedding_model)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    @property
    def name(self) -> str:
        return self._name

    def generate(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = (
            "Use provided context to answer user's query.\n\n"
            f"Context:\n{context}\n\n"
            f"Query: {query}\n\n"
        )

        response = self.llm.invoke(prompt)
        return str(response)


class ThisNIRPipeline(BasePipeline):

    def __init__(
        self,
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        knowledge_graph: KnowledgeGraph,
        use_llm_for_timestamps: bool = True,
        add_history: bool = True,
        max_tokens: int = 1024,
        name: str = "Graph RAG"
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.graph = knowledge_graph
        self.use_llm_for_timestamps = use_llm_for_timestamps
        self.add_history = add_history
        self.max_tokens = max_tokens
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(self, query: str) -> str:
        if self.use_llm_for_timestamps:
            context = form_context_with_llm(
                query=query,
                graph=self.graph,
                llm=self.llm,
                embedding_model=self.embedding_model,
                add_history=self.add_history,
                max_tokens=self.max_tokens
            )
        else:
            context = form_context_without_llm(
                query=query,
                graph=self.graph,
                embedding_model=self.embedding_model,
                add_history=self.add_history,
                max_tokens=self.max_tokens
            )
        answer_plan = generate_plan(query, context, self.llm)
        response = generate_answer_based_on_plan(query, answer_plan, self.llm)
        return str(response)