#This is working chat

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from nir.core.answers_generator import generate_answer_based_on_plan, generate_plan
from nir.core.context_retriever import form_context_with_llm
from nir.data import loader
from nir.graph.builder import update_graph_overall
from nir.graph.graph_construction import update_embeddings, update_graph
from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.llm.manager import ModelManager
from nir.llm.providers import ModelConfig


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
                return 0
            context = form_context_with_llm(query, self.graph, self.instruct_model, self.embedding_model, self.add_history)
            answer_plan = generate_plan(query, context, self.chat_model)
            answer_final = generate_answer_based_on_plan(query, answer_plan, self.chat_model)
            print(answer_final)

            if_update = input("Update graph? (yes or no): ")
            if if_update == "yes":
                update_graph_overall(self.graph, answer_final, self.embedding_model, self.instruct_model)

manager = ModelManager()
model_config = ModelConfig("llama3.2:latest")
chat_model = manager.create_chat_model("basic", "ollama", model_config)
embedding_model = manager.create_embedding_model(name="embeddings", option="ollama", model_name="nomic-embed-text:v1.5")
model_config = ModelConfig(model_name="mistral:7b-instruct", temperature=0)
instruct_model = manager.create_chat_model("graph_extraction", "ollama", model_config)

graph_loaded = NetworkXGraph()
graph_loaded.load("assets/graphs/graph_map_short.json")

chat = Chat(chat_model, instruct_model, embedding_model, graph_loaded)
chat.start_chat()