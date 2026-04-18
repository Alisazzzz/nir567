#This is working chat

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from nir.core.answers_generator import filter_context, generate_answer_based_on_context, generate_answer_based_on_plan, generate_plan
from nir.core.context_retriever import form_context_with_llm, form_context_without_llm
from nir.data import loader
from nir.embedding.vector_store_loader import VectorStoreInfo
from nir.graph.graph_construction import create_embeddings, extract_graph, extract_graph_from_nodes, get_next_chunk_id, update_embeddings, update_graph
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
            context = form_context_without_llm(
                query=query, 
                graph=self.graph, 
                embedding_model=self.embedding_model, 
                add_history=self.add_history
            )
            answer_plan = generate_plan(query, context, self.chat_model)
            print(answer_plan)
            answer_final = generate_answer_based_on_plan(query, answer_plan, context, self.chat_model)
            print(answer_final)

            if_update = input("Update graph? (yes or no): ")
            if if_update == "yes":
                update_graph_overall(self.graph, answer_final, self.embedding_model, self.instruct_model)

manager = ModelManager()

#EMBEDDING MODEL CHOICE
# embedding_model = manager.create_embedding_model(
#     name="embeddings", 
#     option="ollama", 
#     model_name="all-minilm:latest"
# )

# #EMBEDDING MODEL CHOICE
# embedding_model = manager.create_embedding_model(
#     name="embeddings", 
#     option="hf_local", 
#     model_name="ai-forever/ru-en-RoSBERTa"
# )

#EMBEDDING MODEL CHOICE
embedding_model = manager.create_embedding_model(
    name="embeddings", 
    option="hf_local", 
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#CHAT MODEL CHOICE
model_config = ModelConfig(
    model_name="hf.co/VlSav/Vikhr-Nemo-12B-Instruct-R-21-09-24-Q4_K_M-GGUF:latest",
    temperature=0.7,
    max_tokens=1024
)
chat_model = manager.create_chat_model(
    name="basic_chat", 
    option="ollama", 
    config=model_config
)

#GRAPH EXTRACTION MODEL CHOICE
model_config = ModelConfig(
    model_name="hf.co/VlSav/Vikhr-Nemo-12B-Instruct-R-21-09-24-Q4_K_M-GGUF:latest", 
    temperature=0
)
instruct_model = manager.create_chat_model(
    name="graph_extraction", 
    option="ollama", 
    config=model_config)

#GRAPH CHOICE

#this is for graph creation
# data = loader.loadCSV(
#     path="assets/documents/NUCLEAR CRYPT.csv"
# )
# data = loader.loadCSV_withColumns(
#     path="assets/documents/NUCLEAR CRYPT SHORT.csv",
#     columns=["Описание места действия", "Персонаж", "Действие", "Реплика"]
# )
# print(data)

# chunks = loader.to_chunk_unique_id(docs=data, start_chunk_id=0)
# graph = extract_graph_from_nodes(chunks=chunks, llm=instruct_model, embedding_model=embedding_model, graph_class=NetworkXGraph, preserve_all_data=True, language="ru")
# vector_db_info = VectorStoreInfo(
#     type="chromadb",
#     info={ 
#         "name" : "nuclear_crypt_csv_test_from_nodes_short",
#         "path" : "assets/databases/chroma_db"
#     }
# )
# graph.create_vector_db(vector_db_info)
# create_embeddings(graph, graph.get_vector_db(), embedding_model)
# graph.save(filepath="assets/graphs/nuclear_crypt_csv_test_from_nodes_short.json")
# graph.visualize(filepath="assets/outputs/nuclear_crypt_csv_test_from_nodes_short.html")

#this is for graph loading
# graph = NetworkXGraph()
# graph.load(
#     filepath="assets/graphs/gpt_test.json"
# )
# graph.visualize(filepath="assets/outputs/gpt_test.html")

# chat = Chat(
#     chat_model=chat_model, 
#     instruct_model=instruct_model, 
#     embedding_model=embedding_model, 
#     graph=graph,
# )
# chat.set_add_history(False)
# chat.start_chat()

graph = NetworkXGraph()
graph.load(filepath="assets/graphs/notebook_story_from_nodes.json")
query="Create an overview of the first quest, where player playes for an ancient demon, few moments after his transition from rock into a living creature (red cat)."

context = form_context_without_llm(query=query, graph=graph, embedding_model=embedding_model, add_history=False)

# variant 1
print("\n\n")
print("BASIC (ONLY PLAN)")
print("\n")

answer_plan = generate_plan(query, context, chat_model, False, "en")
print("Answer plan")
print(answer_plan)
print("\n")

answer_final = generate_answer_based_on_plan(query, answer_plan, context, chat_model, "en")
print("Final answer")
print(answer_final)
print("\n")


# variant 2
print("\n\n")
print("PLAN WITH THEORY")
print("\n")

answer_plan = generate_plan(query, context, chat_model, True, "en")
print("Answer plan")
print(answer_plan)
print("\n")

answer_final = generate_answer_based_on_plan(query, answer_plan, context, chat_model, "en")
print("Final answer")
print(answer_final)
print("\n")


# variant 3
print("\n\n")
print("ONLY FILTER")
print("\n")

filtered_context = filter_context(query, context, chat_model, "en")
print("Filtered context")
print(filtered_context)
print("\n")

answer_final = generate_answer_based_on_context(query, filtered_context, chat_model, "en")
print("Final answer")
print(answer_final)
print("\n")


# variant 4
print("\n\n")
print("RAW CONTEXT")
print("\n")

answer_final = generate_answer_based_on_context(query, context, chat_model, "en")
print("Final answer")
print(answer_final)
print("\n")