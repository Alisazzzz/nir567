#In case of using NetworkX as graph storage

import networkx as nx
from typing import List, Optional
from langchain_community.graphs.graph_document import GraphDocument
import json
from pyvis.network import Network

from nir.graph.knowledge_graph import KnowledgeGraph
from nir.graph.graph_structures import Node, Edge, State


class NetworkXGraph(KnowledgeGraph):
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_node(self, node: Node) -> None:
        self.graph.add_node(
            node.id,
            data=node.model_dump()
        )

    def add_edge(self, edge: Edge) -> None:
        self.graph.add_edge(
            edge.source,
            edge.target,
            key=edge.relation,
            data=edge.model_dump()
        )

    def get_node_by_name(self, node_name: str) -> Node:
        for node_id, node_attrs in self.graph.nodes(data=True):
            data = node_attrs.get("data")
            if data and data.get("name") == node_name:
                return Node(**data)
        return
    
    def get_node_by_id(self, node_id: str) -> Node:
        if node_id not in self.graph.nodes:
            return
        data = self.graph.nodes[node_id]["data"]
        return Node(**data)

    def get_all_nodes(self) -> List[Node]:
        nodes = []
        for node_id in self.graph.nodes:
            attrs = self.graph.nodes[node_id]
            print(attrs)
            if "data" in attrs:
                nodes.append(Node(**attrs["data"]))
            else:
                continue
        return nodes

    def get_edges_between_nodes(self, source: str, target: str) -> List[Edge]:
        if not self.graph.has_edge(source, target):
            return []
        edges = []
        for key, attrs in self.graph[source][target].items():
            edge_data = attrs["data"]
            edges.append(Edge(**edge_data))
        return edges

    def get_edge_by_id(self, edge_id: str) -> Edge:
        for u, v, attrs in self.graph.edges(data=True):
            if "data" in attrs:
                if attrs["data"]["id"] == edge_id:
                    return Edge(**attrs["data"])

    def get_all_edges(self) -> List[Edge]:
        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            if "data" in attrs:
                edges.append(Edge(**attrs["data"]))
        return edges
    
    def get_neighbours_of_node(self, node_id: str) -> List[Node]:
        connected_ids = set(self.graph.neighbors(node_id)) 
        return [self.get_node_by_id(n) for n in connected_ids]

    def update_node_state(self, node_id: str, new_state: State) -> None:
        node = self.get_node_by_id(node_id)
        changed = False
        for i, state in enumerate(node.states):
            if state.current_description == new_state.current_description:
                node.states[i] = new_state
                changed = False
        if not changed:
            node.states.append(new_state)
        self.graph.nodes[node_id]["data"] = node.model_dump()

    def update_edge_times(self, edge_id: str, new_description: str, time_start_event: Optional[str] = None, time_end_event: Optional[str] = None) -> None:
        edge = self.get_edge_by_id(edge_id)
        if time_start_event:
            edge["data"]["time_start_event"] = time_start_event
        if time_end_event:
            edge["data"]["time_end_event"] = time_end_event
        edge.description = new_description
        self.graph.edges[edge_id]["data"] = edge
        return

    def update_node_full(self, node_id: str, new_info: Node) -> None:
        node = self.get_node_by_id(node_id)
        node.name = new_info.name
        node.base_description = new_info.base_description
        node.base_attributes = new_info.base_attributes
        node.states = new_info.states
        node.chunk_id = new_info.chunk_id
        return
    
    def update_edge_full(self, edge_id: str, new_info: Edge) -> None:
        edge = self.get_edge_by_id(edge_id)
        edge.source = new_info.source
        edge.target = new_info.target
        edge.relation = new_info.relation
        edge.description = edge.description
        edge.weight = new_info.weight
        edge.time_start_event = new_info.time_start_event
        edge.time_end_event = new_info.time_end_event
        edge.chunk_id = new_info.chunk_id
        return
    
    def remove_node(self, node_id: str) -> None:
        return

    def remove_edge(self, edge_id: str) -> None:
        return

    def save(self, filepath: str) -> None:
        nodes_data = []
        for n in self.graph.nodes:
            node_attrs = self.graph.nodes[n]
            if "data" not in node_attrs:
                node_attrs["data"] = {
                    "id": n,
                    "name": n,
                    "type": "item",
                    "description": "Auto-created missing node",
                    "attributes": {},
                    "states": []
                }
            nodes_data.append(node_attrs["data"])
        edges_data = []
        for u, v, attrs in self.graph.edges(data=True):
            if "data" not in attrs:            
                continue
            edges_data.append(attrs["data"])
        data = {"nodes": nodes_data, "edges": edges_data}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str) -> None:
        self.graph.clear()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for node_dict in data["nodes"]:
            node = Node(**node_dict)
            self.add_node(node)
        for edge_dict in data["edges"]:
            edge = Edge(**edge_dict)
            self.add_edge(edge)

    def visualize(self, filepath: str):

        TYPE_COLORS = {
            "character": "#ff6b6b",
            "group": "#ffa94d",
            "location": "#4dabf7",
            "environment_element": "#69db7c",
            "item": "#f783ac",
            "event": "#9775fa",
        }
        net = Network(
            height="800px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            directed=self.graph.is_directed()
        )
        net.barnes_hut(
            gravity=-30000,
            central_gravity=0.3,
            spring_length=150,
            spring_strength=0.002,
            damping=0.9
        )
        for node, attr in self.graph.nodes(data=True):
            data = attr["data"]
            node_type = data.get("type", "unknown")
            color = TYPE_COLORS.get(node_type, "#ced4da")
            net.add_node(
                n_id=str(node),
                label=str(node),
                color=color,
                data=data 
            )
        for u, v, data in self.graph.edges(data=True):
            net.add_edge(
                str(u),
                str(v),
                data=data
            )
        net.write_html(filepath)
        with open(filepath, "r", encoding="utf8") as f:
            html = f.read()

        custom_js = r"""
            <style>
            #infoPanel {
                position: fixed;
                top: 20px;
                right: 20px;
                max-width: 600px;
                background: #1e1e1e;
                color: white;
                padding: 16px;
                border-radius: 8px;
                border: 1px solid #444;
                font-family: Consolas, monospace;
                font-size: 13px;
                display: none;
                z-index: 9999;
                white-space: pre-wrap;
            }
            #infoPanel h3 {
                margin: 0 0 10px 0;
                font-size: 15px;
            }
            </style>
            <div id="infoPanel"></div>
            <script>
            function showPanel(title, data) {
                const panel = document.getElementById("infoPanel");
                panel.style.display = "block";
                panel.innerHTML = "<h3>" + title + "</h3>" +
                                "<pre>" + JSON.stringify(data, null, 2) + "</pre>";
            }

            function hidePanel() {
                document.getElementById("infoPanel").style.display = "none";
            }
            setTimeout(() => {
                if (!window.network) return;
                network.on("click", function (params) {
                    if (params.nodes.length === 0 && params.edges.length === 0) {
                        hidePanel();
                        return;
                    }
                    if (params.nodes.length === 1) {
                        const nodeId = params.nodes[0];
                        const node = network.body.data.nodes.get(nodeId);
                        showPanel("Node: " + nodeId, node.data || {});
                        return;
                    }
                    if (params.edges.length === 1) {
                        const edgeId = params.edges[0];
                        const edge = network.body.data.edges.get(edgeId);
                        showPanel("Edge: " + edge.from + " → " + edge.to, edge.data || {});
                        return;
                    }
                });

            }, 300);
            </script>
        """
        html = html.replace("</body>", custom_js + "\n</body>")
        with open(filepath, "w", encoding="utf8") as f:
            f.write(html)
        return

