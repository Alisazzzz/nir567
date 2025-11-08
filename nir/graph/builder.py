#This file is running for graph creation

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

# manager = ModelManager()
# embedding_model = manager.create_embedding_model(name="embeddings", option="ollama", model_name="nomic-embed-text:v1.5")

# vector_store = ChromaVectorStore(
#     collection_name="ali_baba_graph",
#     persist_directory="assets/databases/chroma_db"
# )

# tools.create_embeddings_from_graph(
#     graph=graph_loaded,
#     vector_store=vector_store,
#     embedding_model=embedding_model
# )

# manager = ModelManager()
# model_config = ModelConfig(model_name="mistral:7b-instruct-q2_K", temperature=0)
# manager.create_chat_model("graph_extraction", "ollama", model_config)
# llm = manager.get_chat_model("graph_extraction")

# graph_extractor = GraphExtractor(llm=llm, graph_class=NetworkXGraph, embedder=embedding_model)

# data = loader.loadTXT("assets/documents/ali baba, or the forty thieves.txt")
# chunks = loader.to_chunk(data)

# graph = graph_extractor.extract_graph(chunks)
# graph.save("assets/graphs/graph.json")
# graph.visualize()

import requests, subprocess, time

def ensure_coref_server():
    try:
        requests.get("http://127.0.0.1:8008/resolve", timeout=2)
        print("coref_server уже запущен.")
    except:
        print("Запускаю coref_server...")
        subprocess.Popen(["python", "../python3.8-server/setup.py"])
        time.sleep(10)  # подождать инициализацию

ensure_coref_server()