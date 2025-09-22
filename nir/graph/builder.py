#This file is running for graph creation

from nir.data import loader
from nir.graph import tools
from nir.graph.graph_storages import networkx_graph
from langchain_ollama.llms import OllamaLLM


llm = OllamaLLM(model="llama3.1:8b", temperature=0)

data = loader.loadTXT("assets/documents/ali baba, or the forty thieves.txt")
chunks = loader.to_chunk(data)
graph = tools.extract_graph(chunks, llm)

networkx_graph.create_networkX_graph(graph)