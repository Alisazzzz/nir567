#This file is running for graph creation

from nir.data import loader
from nir.graph.graph_construction import extract_graph, update_graph, create_embeddings, update_embeddings
from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.graph.knowledge_graph import KnowledgeGraph

from nir.llm.providers import ModelConfig
from nir.llm.manager import ModelManager

from nir.embedding.vector_storages.chroma_db import ChromaVectorStore

manager = ModelManager()
embedding_model = manager.create_embedding_model(name="embeddings", option="ollama", model_name="nomic-embed-text:v1.5")
model_config = ModelConfig(model_name="mistral:7b-instruct", temperature=0)
manager.create_chat_model("graph_extraction", "ollama", model_config)
llm = manager.get_chat_model("graph_extraction")

def get_next_chunk_id(graph: KnowledgeGraph) -> int:
    all_nodes = graph.get_all_nodes()
    all_edges = graph.get_all_edges()
    max_node_id = max([max(n.chunk_id) if n.chunk_id else 0 for n in all_nodes], default=-1)
    max_edge_id = max([e.chunk_id if isinstance(e.chunk_id, int) else max(e.chunk_id, default=0) for e in all_edges], default=-1)
    return max(max_node_id, max_edge_id) + 1

vector_store = ChromaVectorStore(collection_name="lost_map_short", persist_directory="assets/databases/chroma_db")

# data = loader.loadTXT("assets/documents/very short.txt")
# chunks = loader.to_chunk_unique_id(docs=data, start_chunk_id=0)
# graph = extract_graph(
#     chunks=chunks,
#     llm=llm,
#     embedding_model=embedding_model,
#     graph_class=NetworkXGraph)
# graph.save("assets/graphs/graph_map_short.json")
# create_embeddings(graph, vector_store, embedding_model)
# embed_query = embedding_model.embed_query("Who is Mira Stone?")
# print(vector_store.search(embed_query))
# graph.visualize()

graph_loaded = NetworkXGraph()
graph_loaded.load("assets/graphs/graph_map_short.json")
greatest_id = get_next_chunk_id(graph_loaded)
data = loader.loadTXT("assets/documents/very short update.txt")
chunks = loader.to_chunk_unique_id(docs=data, start_chunk_id=greatest_id)
update_graph(chunks, llm, embedding_model, graph_loaded)
graph_loaded.save("assets/graphs/graph_map_short.json")
update_embeddings(graph_loaded, vector_store, embedding_model)
embed_query = embedding_model.embed_query("Who is Elias?")
print(vector_store.search(embed_query))
graph_loaded.visualize()




