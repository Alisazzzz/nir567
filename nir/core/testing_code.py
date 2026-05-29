#This is file for code testing without application

from nir.core.answers_generator import generate_answer_based_on_plan, generate_plan
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
embedding_model = manager.get_embedding_model(name="embeddings_fast")

#GRAPH EXTRACTION MODEL CHOICE
# model_config = ModelConfig(model_name="hf.co/VlSav/Vikhr-Nemo-12B-Instruct-R-21-09-24-Q4_K_M-GGUF:latest", temperature=0.0)
# instruct_model = manager.create_chat_model(name="graph_extraction_local", option="ollama", config=model_config)

# model_config = ModelConfig(model_name="mistral-medium-2505", temperature=0.0)
# instruct_model = manager.create_chat_model(name="graph_extraction_remote", option="mistralai", config=model_config, api_info="h6OQzSN4XVNshjRDx7Tj7md01VXSPH4U")
instruct_model = manager.get_chat_model("graph_extraction_remote")

model_config = ModelConfig(model_name="mistral-medium-2505", temperature=0.85)
chat_model = manager.create_chat_model(name="generation_remote", option="mistralai", config=model_config, api_info="h6OQzSN4XVNshjRDx7Tj7md01VXSPH4U")

#GRAPH CREATION
# data = loader.loadPDF(path="assets/documents/generation_tests/nuclear_crypt_info_english.pdf")
# chunks = loader.to_chunk_unique_id(docs=data, start_chunk_id=0, chunk_size=1500, chunk_overlap=150)
# graph = extract_graph_from_nodes(chunks=chunks, llm=instruct_model, embedding_model=embedding_model, graph_class=NetworkXGraph, preserve_all_data=True, language="en", need_pause=True)
# vector_db_info = VectorStoreInfo(
#     type="chromadb",
#     info={ 
#         "name" : "nuclear_crypt_info_english",
#         "path" : "assets/databases/chroma_db"
#     }
# )
# graph.create_vector_db(vector_db_info)
# create_embeddings(graph, graph.get_vector_db(), embedding_model)
# graph.save(path="assets/graphs/nuclear_crypt_info_graph.json")
# GraphWindow(graph, "assets/graphs/nuclear_crypt_info_graph.json", instruct_model, embedding_model).run()

#GRAPH SELECTION
graph = NetworkXGraph()
graph.load_and_create_vector_db("assets/graphs/nuclear_crypt_info_graph.json")

query = "Create a small robotic companion for Protagonist (Alex). This companion is made by Master and has a form of an animal, but not ordinary animal, something unusual. Describe very shortly its appearance, things it likes and how it helps Protagonist."
context = form_context_with_llm(query, graph, instruct_model, embedding_model, "en")
print("CONTEXT")
print(context)

plan = generate_plan(query, context, chat_model, False, "en")
print("PLAN")
print(plan)

answer = generate_answer_based_on_plan(query, plan, context, chat_model, "en")
print("ANSWER")
print(answer)
