#This is working chat

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from nir.core.answers_generator import generate_answer_based_on_plan, generate_plan
from nir.core.context_retriever import form_context_with_llm
from nir.data import loader
from nir.embedding.vector_store_loader import VectorStoreInfo
from nir.graph.graph_construction import create_embeddings, extract_graph, get_next_chunk_id, update_embeddings, update_graph
from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.llm.manager import ModelManager
from nir.llm.providers import ModelConfig

def update_graph_overall(graph: KnowledgeGraph, text: str, embedding_model: Embeddings, llm: BaseLanguageModel) -> None:
    greatest_id = get_next_chunk_id(graph)
    chunks = loader.to_chunk_unique_id(docs=text, start_chunk_id=greatest_id)
    update_graph(chunks, llm, embedding_model, graph)
    graph.save("assets/graphs/graph_map_short.json")
    update_embeddings(graph, graph.get_vector_db(), embedding_model)

class Chat():
    def __init__(
            self, 
            chat_model: BaseLanguageModel,
            instruct_model: BaseLanguageModel,
            embedding_model: Embeddings,
            graph: KnowledgeGraph
    ) -> None:
        self.add_history = True
        self.current_context = ""
        self.chat_model = chat_model
        self.instruct_model = instruct_model
        self.embedding_model = embedding_model
        self.graph = graph

    def set_add_history(self, new_value: bool) -> None:
        self.add_history = new_value

    def start_chat(self) -> int:
        while True:
            query = input("Ask your question (q to quit): ")
            if query == "q":
                graph.save(filepath="assets/graphs/graph_map_short.json")
                return 0
            context = form_context_with_llm(query, self.graph, self.instruct_model, self.embedding_model, self.add_history)
            answer_plan = generate_plan(query, context, self.chat_model)
            answer_final = generate_answer_based_on_plan(query, answer_plan, self.chat_model)
            print(answer_final)

            if_update = input("Update graph? (yes or no): ")
            if if_update == "yes":
                update_graph_overall(self.graph, answer_final, self.embedding_model, self.instruct_model)

manager = ModelManager()

#EMBEDDING MODEL CHOICE
embedding_model = manager.create_embedding_model(
    name="embeddings", 
    option="ollama", 
    model_name="nomic-embed-text:v1.5"
)

#CHAT MODEL CHOICE
model_config = ModelConfig(
    model_name="llama3.2:latest",
    temperature=0.8
)
chat_model = manager.create_chat_model(
    name="basic_chat", 
    option="ollama", 
    config=model_config
)

#GRAPH EXTRACTION MODEL CHOICE
model_config = ModelConfig(
    model_name="mistral:7b-instruct", 
    temperature=0
)
instruct_model = manager.create_chat_model(
    name="graph_extraction", 
    option="ollama", 
    config=model_config)

#GRAPH CHOICE

#this is for graph creation
data = loader.loadTXT(
    path="assets/documents/very short.txt"
)
chunks = loader.to_chunk_unique_id(docs=data, start_chunk_id=0)
graph = extract_graph(chunks=chunks, llm=instruct_model, embedding_model=embedding_model, graph_class=NetworkXGraph)
vector_db_info = VectorStoreInfo(
    type="chromadb",
    info={ 
        "name" : "very_short",
        "path" : "assets/databases/chroma_db"
    }
)
graph.create_vector_db(vector_db_info)
create_embeddings(graph, graph.get_vector_db(), embedding_model)
graph.save(filepath="assets/graphs/graph_very_short.json")
graph.visualize(filepath="assets/outputs/very_short_graph.html")

#this is for graph loading
# graph = NetworkXGraph()
# graph.load(
#     filepath="assets/graphs/graph_map_short.json"
# )
# print(graph.get_vector_db())
# data = loader.loadTXT(
#     path="assets/documents/very short.txt"
# )
# update_graph_overall(graph, data, embedding_model, instruct_model)

# chat = Chat(chat_model, instruct_model, embedding_model, graph)
# chat.start_chat()