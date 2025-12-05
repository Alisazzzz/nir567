#This file is running for graph creation

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from nir.data import loader
from nir.graph.graph_construction import extract_graph, update_graph, create_embeddings, update_embeddings
from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.graph.knowledge_graph import KnowledgeGraph

from nir.llm.providers import ModelConfig
from nir.llm.manager import ModelManager

from nir.embedding.vector_storages.chroma_db import ChromaVectorStore

# manager = ModelManager()
# embedding_model = manager.create_embedding_model(name="embeddings", option="ollama", model_name="nomic-embed-text:v1.5")
# model_config = ModelConfig(model_name="mistral:7b-instruct", temperature=0)
# manager.create_chat_model("graph_extraction", "ollama", model_config)
# llm = manager.get_chat_model("graph_extraction")

def get_next_chunk_id(graph: KnowledgeGraph) -> int:
    all_nodes = graph.get_all_nodes()
    all_edges = graph.get_all_edges()
    max_node_id = max([max(n.chunk_id) if n.chunk_id else 0 for n in all_nodes], default=-1)
    max_edge_id = max([e.chunk_id if isinstance(e.chunk_id, int) else max(e.chunk_id, default=0) for e in all_edges], default=-1)
    return max(max_node_id, max_edge_id) + 1

vector_store = ChromaVectorStore(collection_name="map_short", persist_directory="assets/databases/chroma_db")

def update_graph_overall(graph: KnowledgeGraph, text: str, embedding_model: Embeddings, llm: BaseLanguageModel) -> None:
    greatest_id = get_next_chunk_id(graph)
    data = loader.convertFromString(text)
    chunks = loader.to_chunk_unique_id(docs=data, start_chunk_id=greatest_id)
    update_graph(chunks, llm, embedding_model, graph)
    graph.save("assets/graphs/graph_map_short.json")
    update_embeddings(graph, vector_store, embedding_model)



# data = loader.loadTXT("assets/documents/script with events.txt")
# chunks = loader.to_chunk_unique_id(docs=data, start_chunk_id=0)
# graph = extract_graph(
#     chunks=chunks,
#     llm=llm,
#     embedding_model=embedding_model,
#     graph_class=NetworkXGraph)
# graph.save("assets/graphs/graph_script_short.json")
# create_embeddings(graph, vector_store, embedding_model)
# embed_query = embedding_model.embed_query("Who is Lira?")
# print(vector_store.search(embed_query))
# graph.visualize()

# graph_loaded = NetworkXGraph()
# graph_loaded.load("assets/graphs/graph_script_short.json")
# greatest_id = get_next_chunk_id(graph_loaded)
# data = loader.loadTXT("assets/documents/script with events update.txt")
# chunks = loader.to_chunk_unique_id(docs=data, start_chunk_id=greatest_id)
# update_graph(chunks, llm, embedding_model, graph_loaded)
# graph_loaded.save("assets/graphs/graph_script_short.json")
# update_embeddings(graph_loaded, vector_store, embedding_model)
# embed_query = embedding_model.embed_query("Who is Kael?")
# print(vector_store.search(embed_query))
# graph_loaded.visualize()




