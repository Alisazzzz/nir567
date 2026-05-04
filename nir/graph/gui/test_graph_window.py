from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.graph.gui.graph_window import GraphWindow

g = NetworkXGraph()
g.load("assets/graphs/leisure_suit_larry.json")

GraphWindow(g).run()