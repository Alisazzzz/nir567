#This is file for code testing without application

from nir.core.context_retriever import form_context_with_llm
from nir.data import loader
from nir.embedding.vector_store_loader import VectorStoreInfo
from nir.graph.graph_construction import create_embeddings, extract_graph, extract_graph_from_nodes
from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.llm.manager import ModelManager
from nir.llm.providers import ModelConfig
from nir.graph.gui.graph_window import GraphWindow

manager = ModelManager()

#EMBEDDING MODEL CHOICE
embedding_model = manager.get_embedding_model(name="embeddings_rus")

#GRAPH EXTRACTION MODEL CHOICE
# model_config = ModelConfig(model_name="hf.co/VlSav/Vikhr-Nemo-12B-Instruct-R-21-09-24-Q4_K_M-GGUF:latest", temperature=0.0)
# instruct_model = manager.create_chat_model(name="graph_extraction_local", option="ollama", config=model_config)

# model_config = ModelConfig(model_name="mistral-medium-2505", temperature=0.0)
# instruct_model = manager.create_chat_model(name="graph_extraction_remote", option="mistralai", config=model_config, api_info="sA9z6RzFyAuH3jqCesal8qZihuGS9BUi")

instruct_model = manager.get_chat_model("graph_extraction_remote")

#GRAPH CREATION
# data = loader.loadPDF(path="assets/documents/generation_tests/nuclear_crypt_info.pdf")
# chunks = loader.to_chunk_unique_id(docs=data, start_chunk_id=0, chunk_size=1000, chunk_overlap=100)
# graph = extract_graph_from_nodes(chunks=chunks, llm=instruct_model, embedding_model=embedding_model, graph_class=NetworkXGraph, preserve_all_data=True, language="ru", need_pause=True)
# vector_db_info = VectorStoreInfo(
#     type="chromadb",
#     info={ 
#         "name" : "nuclear_crypt_info",
#         "path" : "assets/databases/chroma_db"
#     }
# )
# graph.create_vector_db(vector_db_info)
# create_embeddings(graph, graph.get_vector_db(), embedding_model)
# graph.save(path="assets/graphs/nuclear_crypt_info_from_nodes.json")
# GraphWindow(graph, "assets/graphs/nuclear_crypt_info_from_nodes.json", instruct_model, embedding_model).run()

#GRAPH SELECTION
graph = NetworkXGraph()
graph.load("assets/graphs/graph_extraction_tests/nuclear_crypt_synopsis_graph_big_from_nodes.json")

query = "Опиши персонажа-ученого, друга для Главного Героя, находящегося в подводной лаборатории"
context = form_context_with_llm(query, graph, instruct_model, embedding_model, "ru")
print(context)

query = "Опиши персонажа-ученого, друга для Главного Героя, до чрезвычайной ситуации с Валладием-5 на подводной лаборатории"
context = form_context_with_llm(query, graph, instruct_model, embedding_model, "ru")
print(context)

GraphWindow(graph, "assets/graphs/graph_extraction_tests/nuclear_crypt_synopsis_graph_big_from_nodes.json", instruct_model, embedding_model).run()