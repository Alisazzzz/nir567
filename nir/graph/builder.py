#This file is running for graph creation

from typing import List
from fastcoref import FCoref

from nir.data import loader
from nir.graph import tools
from nir.graph.graph_extractor import GraphExtractor
from nir.graph.graph_storages.networkx_graph import NetworkXGraph

from nir.llm.providers import ModelConfig
from nir.llm.manager import ModelManager

from nir.embedding.vector_storages.chroma_db import ChromaVectorStore

# manager = ModelManager()
# model_config = ModelConfig(model_name="mistral:7b-instruct-q2_K", temperature=0)
# manager.create_chat_model("graph_extraction", "ollama", model_config)
# llm = manager.get_chat_model("graph_extraction")

# data = loader.loadTXT("assets/documents/ali baba, or the forty thieves.txt")
# chunks = loader.to_chunk(data)
# graph = tools.extract_graph(chunks, llm)
# graph.save("assets/graphs/graph.json")
# graph.visualize()

# graph_loaded = networkx_graph.NetworkXGraph()
# graph_loaded.load("assets/graphs/graph.json")
# # graph_loaded.visualize()

manager = ModelManager()
embedding_model = manager.create_embedding_model(name="embeddings", option="ollama", model_name="nomic-embed-text:v1.5")

# vector_store = ChromaVectorStore(
#     collection_name="ali_baba_graph",
#     persist_directory="assets/databases/chroma_db"
# )

# tools.create_embeddings_from_graph(
#     graph=graph_loaded,
#     vector_store=vector_store,
#     embedding_model=embedding_model
# )

manager = ModelManager()
model_config = ModelConfig(model_name="mistral:7b-instruct", temperature=0)
manager.create_chat_model("graph_extraction", "ollama", model_config)
llm = manager.get_chat_model("graph_extraction")

def load_txt_as_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content
def resolve_coreference(chunk_text: str) -> List[List[str]]:
    corefres_model = FCoref(device='cuda:0')
    entities = corefres_model.predict(texts=[chunk_text])
    return entities[0].get_clusters()
    
file_path = "assets/documents/lost map text.txt"
text = load_txt_as_string(file_path)
entities = resolve_coreference(text)
graph_extractor = GraphExtractor(llm=llm, graph_class=NetworkXGraph, embedding_model=embedding_model, coreference_list=entities)

data = loader.loadTXT("assets/documents/lost map text.txt")
chunks = loader.to_chunk(data)

# graph = NetworkXGraph()
# graph_extractor._extract_entities(chunks, graph)

# graph.save("assets/graphs/graph_map.json")
# graph.visualize()

graph_loaded = NetworkXGraph()
graph_loaded.load("assets/graphs/graph_map.json")
graph_extractor._extract_events(chunks, graph_loaded)
graph_loaded.save("assets/graphs/graph_map.json")
graph_loaded.visualize()

# import requests, subprocess, time

# def ensure_coref_server():
#     try:
#         requests.get("http://127.0.0.1:8008/resolve", timeout=2)
#         print("coref_server уже запущен.")
#     except:
#         print("Запускаю coref_server...")
#         subprocess.Popen(["python", "../python3.8-server/setup.py"])
#         time.sleep(10)  # подождать инициализацию

# ensure_coref_server()