from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.graph.gui.graph_window import GraphWindow
from nir.llm.manager import ModelManager
from nir.llm.providers import ModelConfig

g = NetworkXGraph()
g.load("assets/graphs/leisure_suit_larry.json")

manager = ModelManager()

model_config = ModelConfig(
    model_name="hf.co/VlSav/Vikhr-Nemo-12B-Instruct-R-21-09-24-Q4_K_M-GGUF:latest", 
    temperature=0.0
)
instruct_model = manager.create_chat_model(
    name="graph_extraction", 
    option="ollama", 
    config=model_config)

embedding_model = manager.get_embedding_model(g.get_embedding_model())
GraphWindow(g, "assets/graphs/leisure_suit_larry.json", instruct_model, embedding_model).run()