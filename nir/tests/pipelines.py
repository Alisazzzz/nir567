#All pipelines for testing are here



#--------------------------
#---------imports----------
#--------------------------

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from nir.core.answers_generator import filter_context, generate_answer_based_on_context, generate_answer_based_on_plan, generate_plan
from nir.graph.graph_construction import extract_graph, extract_graph_from_nodes
from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.core.context_retriever import form_context_with_llm, form_context_without_llm
from nir.llm.manager import ModelManager



#---------------------------------------------------------
#---------abstract pipeline class for generation----------
#---------------------------------------------------------

class BasePipeline(ABC):

    @abstractmethod
    def generate(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass



#--------------------------------------------------
#---------pipeline classes for generation----------
#--------------------------------------------------

class BasicLLMPipeline(BasePipeline):

    def __init__(self, llm: BaseLanguageModel, name="Basic LLM"):
        self.llm = llm
        self._name = name

    @property
    def name(self):
        return self._name

    def generate(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        language = data.get("language", "en")
        context = data.get("context", "")
        if language == "ru":
            prompt = f"Используя данный контекст, дай ответ на запрос.\nКонтекст:\n{context}\nЗапрос:\n{query}"
        else:
            prompt = f"Use provided context to answer user's query.\nContext:\n{context}\nQuery:\n{query}"
        response = self.llm.invoke(prompt)
        return {
            "context": context,
            "answer": str(response)
        }

class StandardRAGPipeline(BasePipeline):

    def __init__(self, llm: BaseLanguageModel, name="Standard RAG"):
        self.llm = llm
        self._name = name

    @property
    def name(self):
        return self._name

    def generate(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        language = data.get("language", "en")
        retriever = data.get("retriever", None)

        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        if language == "ru":
            prompt = f"Используя данный контекст, дай ответ на запрос.\nКонтекст:\n{context}\nЗапрос:\n{query}"
        else:
            prompt = f"Use provided context to answer user's query.\nContext:\n{context}\nQuery:\n{query}"

        response = self.llm.invoke(prompt)
        return {
            "context": context,
            "answer": str(response)
        }

manager = ModelManager()
this_file_embedding_model = manager.create_embedding_model(
        name="embeddings", 
        option="hf_local", 
        model_name="sentence-transformers/all-MiniLM-L6-v2"
)

class ThisNIRBasicPipeline_withLLM(BasePipeline):

    def __init__(self, llm, name="Basic (only graph) with time"):
        self.llm = llm
        self._name = name

    @property
    def name(self):
        return self._name

    def generate(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        language = data.get("language", "en")
        graph = data.get("graph", None)
        embedding_model = data.get("embedding_model", None)

        add_history = data.get("add_history", False)

        context = form_context_with_llm(
            query=query,
            graph=graph,
            llm=self.llm,
            language=language,
            embedding_model=this_file_embedding_model,
            add_history=add_history
        )

        plan = generate_plan(query, context, self.llm, False, language)
        response = generate_answer_based_on_plan(query, plan, context, self.llm, language)

        return {
            "answer": str(response),
            "context": context
        }

class ThisNIRBasicPipeline_withoutLLM(BasePipeline):

    def __init__(self, llm, name="Basic (only graph) without time"):
        self.llm = llm
        self._name = name

    @property
    def name(self):
        return self._name

    def generate(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        language = data.get("language", "en")
        graph = data.get("graph", None)
        embedding_model = data.get("embedding_model", None)
        add_history = data.get("add_history", False)

        context = form_context_without_llm(
            query=query,
            graph=graph,
            embedding_model=embedding_model,
            add_history=add_history
        )

        plan = generate_plan(query, context, self.llm, False, language)
        response = generate_answer_based_on_plan(query, plan, context, self.llm, language)

        return {
            "answer": str(response),
            "context": context
        }


class ThisNIRTheoreticalPipeline_withLLM(BasePipeline):

    def __init__(self, llm, name="Plan with theory with time"):
        self.llm = llm
        self._name = name

    @property
    def name(self):
        return self._name

    def generate(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        language = data.get("language", "en")
        graph = data.get("graph", None)
        embedding_model = data.get("embedding_model", None)
        add_history = data.get("add_history", False)

        context = form_context_with_llm(
            query=query,
            graph=graph,
            llm=self.llm,
            language=language,
            embedding_model=embedding_model,
            add_history=add_history
        )

        plan = generate_plan(query, context, self.llm, True, language)
        response = generate_answer_based_on_plan(query, plan, context, self.llm, language)

        return {
            "answer": str(response),
            "context": context
        }

class ThisNIRTheoreticalPipeline_withoutLLM(BasePipeline):

    def __init__(self, llm, name="Plan with theory without time"):
        self.llm = llm
        self._name = name

    @property
    def name(self):
        return self._name

    def generate(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        language = data.get("language", "en")
        graph = data.get("graph", None)
        embedding_model = data.get("embedding_model", None)
        add_history = data.get("add_history", False)

        context = form_context_without_llm(
            query=query,
            graph=graph,
            embedding_model=embedding_model,
            add_history=add_history,
            max_tokens=512
        )

        plan = generate_plan(query, context, self.llm, True, language)
        response = generate_answer_based_on_plan(query, plan, context, self.llm, language)

        return {
            "answer": str(response),
            "context": context
        }


class ThisNIROnlyFilteringPipeline_withLLM(BasePipeline):

    def __init__(self, llm, name="Only filter context with time"):
        self.llm = llm
        self._name = name

    @property
    def name(self):
        return self._name

    def generate(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        language = data.get("language", "en")
        graph = data.get("graph", None)
        embedding_model = data.get("embedding_model", None)
        add_history = data.get("add_history", False)

        context = form_context_with_llm(
            query=query,
            graph=graph,
            llm=self.llm,
            language=language,
            embedding_model=embedding_model,
            add_history=add_history
        )

        filtered_context = filter_context(query, context, self.llm, language)
        response = generate_answer_based_on_context(query, filtered_context, self.llm, language)

        return {
            "answer": str(response),
            "context": filtered_context
        }

class ThisNIROnlyFilteringPipeline_withoutLLM(BasePipeline):

    def __init__(self, llm, name="Only filter context without time"):
        self.llm = llm
        self._name = name

    @property
    def name(self):
        return self._name

    def generate(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        language = data.get("language", "en")
        graph = data.get("graph", None)
        embedding_model = data.get("embedding_model", None)
        add_history = data.get("add_history", False)

        context = form_context_without_llm(
            query=query,
            graph=graph,
            embedding_model=embedding_model,
            add_history=add_history
        )

        filtered_context = filter_context(query, context, self.llm, language)
        response = generate_answer_based_on_context(query, filtered_context, self.llm, language)

        return {
            "answer": str(response),
            "context": filtered_context
        }
    

class ThisNIRRawContextPipeline_withLLM(BasePipeline):

    def __init__(self, llm, name="Raw context with time"):
        self.llm = llm
        self._name = name

    @property
    def name(self):
        return self._name

    def generate(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        language = data.get("language", "en")
        graph = data.get("graph", None)
        embedding_model = data.get("embedding_model", None)
        add_history = data.get("add_history", False)

        context = form_context_with_llm(
            query=query,
            graph=graph,
            llm=self.llm,
            language=language,
            embedding_model=embedding_model,
            add_history=add_history
        )
        response = generate_answer_based_on_context(query, context, self.llm, language)
        return {
            "answer": str(response),
            "context": context
        }

class ThisNIRRawContextPipeline_withoutLLM(BasePipeline):

    def __init__(self, llm, name="Raw context without time"):
        self.llm = llm
        self._name = name

    @property
    def name(self):
        return self._name

    def generate(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        language = data.get("language", "en")
        graph = data.get("graph", None)
        embedding_model = data.get("embedding_model", None)
        add_history = data.get("add_history", False)

        context = form_context_without_llm(
            query=query,
            graph=graph,
            embedding_model=embedding_model,
            add_history=add_history
        )
        response = generate_answer_based_on_context(query, context, self.llm, language)
        return {
            "answer": str(response),
            "context": context
        }



#---------------------------------------------------------------
#---------abstract pipeline class for graph extraction----------
#---------------------------------------------------------------

class BaseGraphPipeline(ABC):

    @abstractmethod
    def extract_graph(self, chunks: List[Document], embedding_model: Embeddings, language: str = "en") -> KnowledgeGraph:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


#--------------------------------------------------------
#---------pipeline classes for graph extraction----------
#--------------------------------------------------------

class GraphPipelineBasic(BaseGraphPipeline):

    def __init__(        
        self,     
        llm: BaseLanguageModel,     
        graph_class: KnowledgeGraph = NetworkXGraph,
        preserve_all_data: bool = True,
        name: str = "Graph extraction basic"
    ):
        self.llm = llm
        self.graph_class = graph_class
        self.preserve_all_data = preserve_all_data
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def extract_graph(self, chunks: List[Document], embedding_model: Embeddings, language: str = "en") -> KnowledgeGraph:
        result_graph = extract_graph(
            chunks=chunks,
            llm=self.llm,
            embedding_model=embedding_model,
            graph_class=self.graph_class,
            preserve_all_data=self.preserve_all_data,
            language=language
        )
        return result_graph


class GraphPipelineFromNodes(BaseGraphPipeline):

    def __init__(        
        self,     
        llm: BaseLanguageModel,
        graph_class: KnowledgeGraph = NetworkXGraph,
        preserve_all_data: bool = True,
        name: str = "Graph extraction from nodes"
    ):
        self.llm = llm
        self.graph_class = graph_class
        self.preserve_all_data = preserve_all_data
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def extract_graph(self, chunks: List[Document], embedding_model: Embeddings, language: str = "en") -> KnowledgeGraph:
        result_graph = extract_graph_from_nodes(
            chunks=chunks,
            llm=self.llm,
            embedding_model=embedding_model,
            graph_class=self.graph_class,
            preserve_all_data=self.preserve_all_data,
            language=language
        )
        return result_graph
        
