# ==========================================================
# INTERACTIVE GRAPH VISUALIZATION
# UI REWORKED TO MATCH PROVIDED SCREENSHOTS
# (logic preserved, visuals rewritten)
# PART 1 / continue after this block
# ==========================================================

import tkinter as tk
from tkinter import ttk
import networkx as nx
import math
import random

# ----------------------------------------------------------
# COLORS (carefully tuned to screenshot)
# ----------------------------------------------------------

BG_DARK = "#030303"
BG_NODE = "#060606"
BG_PANEL = "#050505"

LINE_GREEN = "#00a84a"
LINE_GREEN_SOFT = "#00863a"
LINE_GREEN_DARK = "#00652b"

TEXT_MAIN = "#00d86a"
TEXT_LIGHT = "#22ff88"
TEXT_DIM = "#0ca84f"
TEXT_FAINT = "#087538"

TEXT_INFO_TITLE = "#17ff7a"
TEXT_INFO_LABEL = "#0e9e49"
TEXT_INFO_VALUE = "#17d96d"

WARN_ORANGE = "#f0aa00"
DANGER_RED = "#ff3030"
MULTI_BLUE = "#00d8ff"

EDGE_COLOR = "#00a648"
EDGE_SELECTED = "#d2ff00"

# ----------------------------------------------------------
# NODE COLORS (as screenshot)
# ----------------------------------------------------------

TYPE_COLORS = {
    "person": "#00ff66",
    "organization": "#00c8ff",
    "location": "#e1bf00",
    "event": "#9a44ff",
    "character": "#00ff66",
    "group": "#00c8ff",
    "environment_element": "#e1bf00",
    "item": "#9a44ff",
    "default": "#aaaaaa"
}

TYPE_LABELS = {
    "person": "PERSON",
    "organization": "ORGANIZATION",
    "location": "LOCATION",
    "event": "EVENT",
    "character": "PERSON",
    "group": "ORGANIZATION",
    "environment_element": "LOCATION",
    "item": "EVENT"
}

TYPE_ZONES = {
    "person": {"x": -350, "y": -250},
    "organization": {"x": 350, "y": -250},
    "location": {"x": -120, "y": 220},
    "event": {"x": 380, "y": 220},
}

NODE_RADIUS = 28
CELL_SIZE = 130

ICON_EDIT = "✎"
ICON_DELETE = "⌫"
ICON_SAVE = "[SAVE]"
ICON_CANCEL = "[CANCEL]"


# ==========================================================
# MAIN CLASS
# ==========================================================

class GraphWindow:

    def __init__(self, graph_storage):
        self.storage = graph_storage
        self.graph = graph_storage.graph

        self.root = tk.Tk()
        self.root.title("Interactive graph visualization")
        self.root.geometry("1600x900")
        self.root.configure(bg=BG_DARK)

        self.zoom = 1.0
        self.pan_offset = [400.0, 220.0]

        self.last_mouse_x = 0
        self.last_mouse_y = 0

        self.is_sorted = False
        self.initial_positions = {}

        self.is_editing = False

        self.selected_nodes = set()
        self.selected_edges = set()

        self.node_positions = {}
        self.screen_positions = {}

        self.node_items = {}
        self.edge_items = {}
        self.edge_labels = {}

        self.dragging_node = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0

        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0

        self._setup_theme()
        self._build_ui()
        self._create_layout()

        self.redraw()
        self.update_info_panel()

    # ======================================================
    # THEME
    # ======================================================

    def _setup_theme(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "TButton",
            background=BG_PANEL,
            foreground=TEXT_MAIN,
            bordercolor=LINE_GREEN,
            lightcolor=LINE_GREEN,
            darkcolor=LINE_GREEN,
            padding=4,
            font=("Consolas", 10)
        )

        style.map(
            "TButton",
            background=[("active", BG_PANEL)],
            foreground=[("active", TEXT_LIGHT)]
        )

    # ======================================================
    # BUILD UI
    # ======================================================

    def _build_ui(self):

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        # LEFT SIDE
        left = tk.Frame(self.root, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew")

        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        # top title
        title = tk.Label(
            left,
            text="$ graph-editor --mode interactive",
            bg=BG_DARK,
            fg=TEXT_MAIN,
            font=("Consolas", 10),
            anchor="w"
        )
        title.grid(row=0, column=0, sticky="ew", padx=8, pady=(4, 0))

        sep = tk.Frame(left, bg=LINE_GREEN, height=1)
        sep.grid(row=1, column=0, sticky="ew")

        # canvas
        self.canvas = tk.Canvas(
            left,
            bg=BG_DARK,
            highlightthickness=0,
            bd=0
        )
        self.canvas.grid(row=2, column=0, sticky="nsew")

        # sort button
        self.sort_btn = tk.Button(
            self.canvas,
            text="[ ↓ SORT_BY_TYPE ]",
            bg=BG_PANEL,
            fg=TEXT_MAIN,
            activebackground=BG_PANEL,
            activeforeground=TEXT_LIGHT,
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=LINE_GREEN,
            font=("Consolas", 9)
        )
        self.sort_btn.config(command=self.toggle_sort)
        self.sort_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-14, y=14)

        # vertical line
        vr = tk.Frame(self.root, width=1, bg=LINE_GREEN)
        vr.grid(row=0, column=1, sticky="ns")

        # RIGHT PANEL
        self.right = tk.Frame(self.root, bg=BG_DARK, width=290)
        self.right.grid(row=0, column=2, sticky="ns")
        self.right.grid_propagate(False)

        self._build_right_panel()

        # binds
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.canvas.bind("<Button-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)

        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_event(120))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_event(-120))

        self.canvas.bind("<Motion>", self._track_mouse)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        self.draw_static_legend()

    # ======================================================
    # RIGHT PANEL
    # ======================================================

    def _build_right_panel(self):

        hdr = tk.Frame(self.right, bg=BG_DARK)
        hdr.pack(fill="x", padx=10, pady=(4, 0))

        tk.Label(
            hdr,
            text="▶ INFO",
            fg=TEXT_MAIN,
            bg=BG_DARK,
            font=("Consolas", 10),
            anchor="w"
        ).pack(anchor="w")

        tk.Frame(hdr, bg=LINE_GREEN, height=1).pack(fill="x", pady=(4, 0))

        # info box
        self.info_wrap = tk.Frame(
            self.right,
            bg=BG_PANEL,
            highlightthickness=1,
            highlightbackground=LINE_GREEN
        )
        self.info_wrap.pack(fill="both", expand=True, padx=10, pady=(8, 8))

        self.info_canvas = tk.Canvas(
            self.info_wrap,
            bg=BG_PANEL,
            highlightthickness=0
        )
        self.info_canvas.pack(side="left", fill="both", expand=True)

        self.info_content = tk.Frame(
            self.info_canvas,
            bg=BG_PANEL
        )

        self.info_canvas.create_window(
            (0, 0),
            window=self.info_content,
            anchor="nw"
        )

        self.info_content.bind(
            "<Configure>",
            lambda e: self.info_canvas.configure(
                scrollregion=self.info_canvas.bbox("all")
            )
        )

        # legend bottom (fixed height so all fits)
        leg = tk.Frame(self.right, bg=BG_DARK)
        leg.pack(fill="x", side="bottom", padx=10, pady=(0, 8))

        tk.Label(
            leg,
            text="▶ LEGEND",
            fg=TEXT_MAIN,
            bg=BG_DARK,
            font=("Consolas", 10),
            anchor="w"
        ).pack(anchor="w")

        self.legend_canvas = tk.Canvas(
            leg,
            bg=BG_DARK,
            height=170,
            highlightthickness=0
        )
        self.legend_canvas.pack(fill="x", pady=(4, 0))

# ===== CONTINUE NEXT MESSAGE =====
    # ======================================================
    # INFO PANEL LOGIC
    # ======================================================

    def clear_info(self):
        for w in self.info_content.winfo_children():
            w.destroy()

    def update_info_panel(self):
        self.clear_info()

        if self.is_editing and len(self.selected_nodes) == 1:
            self._draw_edit_mode()
        elif len(self.selected_nodes) > 1:
            self._draw_multi_select()
        elif len(self.selected_nodes) == 1:
            self._draw_node_info()
        elif len(self.selected_edges) == 1:
            self._draw_edge_info()
        else:
            self._draw_empty_state()

        self.info_canvas.update_idletasks()
        self.info_canvas.configure(
            scrollregion=self.info_canvas.bbox("all")
        )

    # ------------------------------------------------------

    def _draw_empty_state(self):

        box = tk.Frame(
            self.info_content,
            bg=BG_PANEL,
            highlightthickness=1,
            highlightbackground=LINE_GREEN_DARK
        )
        box.pack(fill="x", padx=8, pady=8)

        tk.Label(
            box,
            text="[ NO SELECTION ]",
            bg=BG_PANEL,
            fg=TEXT_DIM,
            font=("Consolas", 9)
        ).pack(pady=(10, 4))

        tk.Label(
            box,
            text="Click node/edge to view",
            bg=BG_PANEL,
            fg=TEXT_FAINT,
            font=("Consolas", 8)
        ).pack(pady=(0, 10))

    # ------------------------------------------------------

    def _draw_pair(self, parent, title, value):
        tk.Label(
            parent,
            text=title,
            bg=BG_PANEL,
            fg=TEXT_INFO_LABEL,
            font=("Consolas", 8)
        ).pack(anchor="w", padx=10, pady=(8, 1))

        tk.Label(
            parent,
            text=value if value else "—",
            bg=BG_PANEL,
            fg=TEXT_INFO_VALUE,
            font=("Consolas", 9),
            justify="left",
            wraplength=230,
            anchor="w"
        ).pack(anchor="w", padx=10)

    # ------------------------------------------------------

    def _icon_button(self, parent, txt, fg, cmd):
        b = tk.Button(
            parent,
            text=txt,
            command=cmd,
            bg=BG_PANEL,
            fg=fg,
            activebackground=BG_PANEL,
            activeforeground=fg,
            relief="flat",
            bd=0,
            font=("Consolas", 10),
            padx=2,
            pady=0,
            cursor="hand2"
        )
        return b

    # ------------------------------------------------------

    def _draw_node_info(self):

        node_id = list(self.selected_nodes)[0]
        node = self.storage.get_node_by_id(node_id)

        ntype = getattr(node, "type", "default")
        color = TYPE_COLORS.get(ntype, TYPE_COLORS["default"])

        # top line
        top = tk.Frame(self.info_content, bg=BG_PANEL)
        top.pack(fill="x", pady=(6, 2), padx=8)

        tk.Label(
            top,
            text=f"[NODE]",
            fg=color,
            bg=BG_PANEL,
            font=("Consolas", 9)
        ).pack(side="left")

        # icon only buttons
        btn_del = self._icon_button(
            top,
            ICON_DELETE,
            DANGER_RED,
            lambda: self._delete_node(node_id)
        )
        btn_del.pack(side="right", padx=(4, 0))

        btn_edit = self._icon_button(
            top,
            ICON_EDIT,
            WARN_ORANGE,
            lambda: self._start_editing(node_id)
        )
        btn_edit.pack(side="right")

        # formatted block exactly like screenshot
        self._draw_pair(self.info_content, "LABEL", node.name)

        desc = getattr(node, "base_description", "")
        self._draw_pair(self.info_content, "DESCRIPTION", desc)

        if hasattr(node, "founded"):
            self._draw_pair(self.info_content, "FOUNDED", str(node.founded))

        if hasattr(node, "industry"):
            self._draw_pair(self.info_content, "INDUSTRY", str(node.industry))

    # ------------------------------------------------------

    def _draw_edge_info(self):

        eid = list(self.selected_edges)[0]
        edge_data = None

        for _, val in self.edge_items.items():
            if val["id"] == eid:
                edge_data = val
                break

        if not edge_data:
            self._draw_empty_state()
            return

        top = tk.Frame(self.info_content, bg=BG_PANEL)
        top.pack(fill="x", pady=(6, 2), padx=8)

        tk.Label(
            top,
            text="[EDGE]",
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            font=("Consolas", 9)
        ).pack(side="left")

        self._icon_button(
            top,
            ICON_DELETE,
            DANGER_RED,
            lambda: self._delete_edge(eid)
        ).pack(side="right")

        self._draw_pair(
            self.info_content,
            "RELATION",
            edge_data.get("relation", "")
        )

    # ------------------------------------------------------

    def _draw_edit_mode(self):

        node_id = list(self.selected_nodes)[0]
        node = self.storage.get_node_by_id(node_id)

        top = tk.Frame(self.info_content, bg=BG_PANEL)
        top.pack(fill="x", pady=(6, 6), padx=8)

        tk.Label(
            top,
            text="[NODE]",
            fg=TEXT_MAIN,
            bg=BG_PANEL,
            font=("Consolas", 9)
        ).pack(side="left")

        self._icon_button(
            top,
            ICON_DELETE,
            DANGER_RED,
            lambda: self._delete_node(node_id)
        ).pack(side="right")

        self._icon_button(
            top,
            ICON_EDIT,
            WARN_ORANGE,
            lambda: None
        ).pack(side="right")

        # LABEL
        tk.Label(
            self.info_content,
            text="LABEL",
            fg=TEXT_INFO_LABEL,
            bg=BG_PANEL,
            font=("Consolas", 8)
        ).pack(anchor="w", padx=10, pady=(4, 2))

        self.edit_name_var = tk.StringVar(value=node.name)

        e1 = tk.Entry(
            self.info_content,
            textvariable=self.edit_name_var,
            bg=BG_DARK,
            fg=WARN_ORANGE,
            insertbackground=WARN_ORANGE,
            relief="flat",
            font=("Consolas", 9)
        )
        e1.pack(fill="x", padx=10, ipady=3)

        e1.config(
            highlightthickness=1,
            highlightbackground=WARN_ORANGE,
            highlightcolor=WARN_ORANGE
        )

        # DESCRIPTION
        tk.Label(
            self.info_content,
            text="DESCRIPTION",
            fg=TEXT_INFO_LABEL,
            bg=BG_PANEL,
            font=("Consolas", 8)
        ).pack(anchor="w", padx=10, pady=(8, 2))

        desc = getattr(node, "base_description", "")

        # IMPORTANT:
        # same size as normal text, no huge 4-line shrink bug
        lines = max(1, desc.count("\n") + 1)

        self.edit_desc_text = tk.Text(
            self.info_content,
            height=lines,
            bg=BG_DARK,
            fg=WARN_ORANGE,
            insertbackground=WARN_ORANGE,
            wrap="word",
            relief="flat",
            font=("Consolas", 9)
        )
        self.edit_desc_text.pack(fill="x", padx=10)
        self.edit_desc_text.insert("1.0", desc)

        self.edit_desc_text.config(
            highlightthickness=1,
            highlightbackground=WARN_ORANGE,
            highlightcolor=WARN_ORANGE
        )

        # buttons
        row = tk.Frame(self.info_content, bg=BG_PANEL)
        row.pack(fill="x", padx=10, pady=(10, 4))

        tk.Button(
            row,
            text="[SAVE]",
            command=lambda: self._save_edit(node_id),
            bg=BG_PANEL,
            fg=TEXT_MAIN,
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=LINE_GREEN,
            font=("Consolas", 9)
        ).pack(side="left", fill="x", expand=True)

        tk.Button(
            row,
            text="[CANCEL]",
            command=self._cancel_edit,
            bg=BG_PANEL,
            fg=DANGER_RED,
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=DANGER_RED,
            font=("Consolas", 9)
        ).pack(side="left", fill="x", expand=True, padx=(6, 0))

# ===== CONTINUE NEXT MESSAGE =====
    # ------------------------------------------------------

    def _draw_multi_select(self):

        top = tk.Frame(self.info_content, bg=BG_PANEL)
        top.pack(fill="x", padx=8, pady=(8, 6))

        tk.Label(
            top,
            text=f"[MULTI_SELECT] {len(self.selected_nodes)} nodes",
            fg=MULTI_BLUE,
            bg=BG_PANEL,
            font=("Consolas", 9)
        ).pack(anchor="w")

        box = tk.Frame(
            self.info_content,
            bg=BG_DARK,
            highlightthickness=1,
            highlightbackground=LINE_GREEN_DARK
        )
        box.pack(fill="x", padx=8, pady=(0, 8))

        for nid in self.selected_nodes:
            node = self.storage.get_node_by_id(nid)

            tk.Label(
                box,
                text=f"• {node.name}",
                fg=TEXT_MAIN,
                bg=BG_DARK,
                anchor="w",
                font=("Consolas", 8)
            ).pack(fill="x", padx=8, pady=2)

        tk.Button(
            self.info_content,
            text="> To MERGE_NODES",
            command=self._merge_nodes,
            bg=BG_PANEL,
            fg=TEXT_MAIN,
            activebackground=BG_PANEL,
            activeforeground=TEXT_LIGHT,
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=LINE_GREEN,
            font=("Consolas", 8)
        ).pack(fill="x", padx=8, pady=(0, 8))

    # ======================================================
    # EDIT ACTIONS
    # ======================================================

    def _start_editing(self, node_id):
        self.is_editing = True
        self.update_info_panel()

    def _cancel_edit(self):
        self.is_editing = False
        self.update_info_panel()

    def _save_edit(self, node_id):

        node = self.storage.get_node_by_id(node_id)

        node.name = self.edit_name_var.get()
        node.base_description = self.edit_desc_text.get(
            "1.0",
            "end-1c"
        )

        self.is_editing = False
        self.redraw()
        self.update_info_panel()

    def _delete_node(self, node_id):
        self.graph.remove_node(node_id)
        self.selected_nodes.clear()
        self.redraw()
        self.update_info_panel()

    def _delete_edge(self, edge_id):
        pass

    def _merge_nodes(self):
        print("merge logic")

    # ======================================================
    # LEGEND
    # ======================================================

    def draw_static_legend(self):

        self.legend_canvas.delete("all")

        y = 10

        order = ["person", "organization", "location", "event"]

        for t in order:

            color = TYPE_COLORS[t]
            label = TYPE_LABELS[t]

            self.legend_canvas.create_oval(
                8, y - 4, 16, y + 4,
                fill=color,
                outline=color
            )

            self.legend_canvas.create_text(
                28,
                y,
                text=label,
                fill=color,
                anchor="w",
                font=("Consolas", 8)
            )

            y += 24

    # ======================================================
    # LAYOUT
    # ======================================================

    def _create_layout(self):

        pos = nx.spring_layout(
            self.graph,
            seed=42,
            k=1.5,
            iterations=50
        )

        for node_id, (x, y) in pos.items():
            self.node_positions[node_id] = [
                800 + x * 500,
                450 + y * 350
            ]

        self.initial_positions = {
            k: v[:] for k, v in self.node_positions.items()
        }

    # ------------------------------------------------------

    def toggle_sort(self):

        if self.is_sorted:
            self.node_positions = {
                k: v[:] for k, v in self.initial_positions.items()
            }
            self.is_sorted = False
            self.sort_btn.config(text="[ ↓ SORT_BY_TYPE ]")

        else:
            self._generate_sorted_layout()
            self.is_sorted = True
            self.sort_btn.config(text="[ ↑ UNSORT ]")

        self.redraw()

    # ------------------------------------------------------

    def _generate_sorted_layout(self):

        groups = {}

        for node_id, attrs in self.graph.nodes(data=True):

            ntype = attrs["data"]["type"]
            groups.setdefault(ntype, []).append(node_id)

        new_pos = {}

        for ntype, nodes in groups.items():

            zone = TYPE_ZONES.get(
                ntype,
                {"x": 0, "y": 0}
            )

            zone_cx = 800 + zone["x"]
            zone_cy = 450 + zone["y"]

            count = len(nodes)

            if count == 0:
                continue

            cols = math.ceil(math.sqrt(count))
            rows = math.ceil(count / cols)

            sw = cols * CELL_SIZE
            sh = rows * CELL_SIZE

            sx = zone_cx - sw / 2
            sy = zone_cy - sh / 2

            shuffled = nodes.copy()
            random.shuffle(shuffled)

            for i, nid in enumerate(shuffled):

                row = i // cols
                col = i % cols

                cx = sx + col * CELL_SIZE + CELL_SIZE / 2
                cy = sy + row * CELL_SIZE + CELL_SIZE / 2

                jitter = CELL_SIZE / 5

                cx += random.uniform(-jitter, jitter)
                cy += random.uniform(-jitter, jitter)

                new_pos[nid] = [cx, cy]

        self.node_positions = new_pos

# ===== CONTINUE NEXT MESSAGE =====

    # ======================================================
    # DRAWING
    # ======================================================

    def redraw(self):

        self.canvas.delete("all")

        self.node_items.clear()
        self.edge_items.clear()
        self.edge_labels.clear()

        self.screen_positions = {}

        for nid, (wx, wy) in self.node_positions.items():
            self.screen_positions[nid] = self.to_screen(wx, wy)

        self.draw_edges()
        self.draw_nodes()
        self.draw_hint()

        self.sort_btn.lift()

    # ------------------------------------------------------

    def to_screen(self, wx, wy):
        return (
            (wx + self.pan_offset[0]) * self.zoom,
            (wy + self.pan_offset[1]) * self.zoom
        )

    def to_world(self, sx, sy):
        return (
            sx / self.zoom - self.pan_offset[0],
            sy / self.zoom - self.pan_offset[1]
        )

    # ======================================================
    # BOTTOM LEFT STATUS
    # ======================================================

    def draw_hint(self):

        txt = (
            f"{int(self.zoom * 100)}%  144k\n"
            "simulation | springgraph | ctrl+clickmulti"
        )

        self.canvas.create_rectangle(
            8, self.canvas.winfo_height() - 44,
            136, self.canvas.winfo_height() - 8,
            outline=LINE_GREEN,
            fill=BG_PANEL,
            width=1
        )

        self.canvas.create_text(
            14,
            self.canvas.winfo_height() - 34,
            text=txt,
            fill=TEXT_DIM,
            anchor="nw",
            font=("Consolas", 6)
        )

    # ======================================================
    # NODES
    # ======================================================

    def draw_nodes(self):

        r = NODE_RADIUS * self.zoom

        for node_id, attrs in self.graph.nodes(data=True):

            data = attrs["data"]

            x, y = self.screen_positions[node_id]

            selected = node_id in self.selected_nodes

            node_color = TYPE_COLORS.get(
                data["type"],
                TYPE_COLORS["default"]
            )

            # IMPORTANT:
            # selected node keeps own color,
            # only yellow outer rings appear
            if selected:

                self.canvas.create_oval(
                    x - r * 1.18, y - r * 1.18,
                    x + r * 1.18, y + r * 1.18,
                    outline="#d9c100",
                    width=max(1, int(1.4 * self.zoom))
                )

                self.canvas.create_oval(
                    x - r * 1.30, y - r * 1.30,
                    x + r * 1.30, y + r * 1.30,
                    outline="#7e6e00",
                    width=1
                )

            oval = self.canvas.create_oval(
                x - r, y - r,
                x + r, y + r,
                fill=BG_NODE,
                outline=node_color,
                width=max(1, int(1.8 * self.zoom))
            )

            self.node_items[oval] = node_id

            if self.zoom >= 0.45:

                txt = self.canvas.create_text(
                    x,
                    y,
                    text=data["name"],
                    fill=node_color,
                    width=70 * self.zoom,
                    justify="center",
                    font=(
                        "Consolas",
                        max(6, int(7 * self.zoom))
                    )
                )

                self.node_items[txt] = node_id

    # ======================================================
    # EDGES
    # ======================================================

    def _get_edge_end_point(self, x1, y1, x2, y2, radius):

        dx = x2 - x1
        dy = y2 - y1

        length = math.hypot(dx, dy)

        if length == 0:
            return x2, y2

        nx = dx / length
        ny = dy / length

        return (
            x2 - nx * radius,
            y2 - ny * radius
        )

    # ------------------------------------------------------

    def draw_edges(self):

        groups = {}

        for u, v, key, attrs in self.graph.edges(keys=True, data=True):

            ed = attrs["data"]

            pair = tuple(sorted([u, v]))

            groups.setdefault(
                pair,
                {"forward": [], "backward": []}
            )

            if ed.get("source") == u:
                groups[pair]["forward"].append((u, v, ed))
            else:
                groups[pair]["backward"].append((v, u, ed))

        for pair, block in groups.items():

            total = (
                len(block["forward"]) +
                len(block["backward"])
            )

            for i, item in enumerate(block["forward"]):
                self._draw_curved_edge(
                    item[0], item[1], item[2],
                    1, i, total
                )

            offset = len(block["forward"])

            for i, item in enumerate(block["backward"]):
                self._draw_curved_edge(
                    item[0], item[1], item[2],
                    -1, offset + i, total
                )

# ===== CONTINUE NEXT MESSAGE =====

    # ------------------------------------------------------

    def _draw_curved_edge(
        self,
        u,
        v,
        edge_data,
        direction=1,
        index=0,
        total_edges=1
    ):

        sx1, sy1 = self.screen_positions[u]
        sx2, sy2 = self.screen_positions[v]

        dx = sx2 - sx1
        dy = sy2 - sy1

        length = math.hypot(dx, dy)
        if length == 0:
            length = 1

        nx = dx / length
        ny = dy / length

        px = -ny
        py = nx

        mx = (sx1 + sx2) / 2
        my = (sy1 + sy2) / 2

        base_curve = 58 * direction * self.zoom
        spread = (
            (index - (total_edges - 1) / 2)
            * 34 * self.zoom
        )

        cx = mx + px * (base_curve + spread)
        cy = my + py * (base_curve + spread)

        r = NODE_RADIUS * self.zoom

        start_x, start_y = self._get_edge_end_point(
            sx2, sy2, sx1, sy1, r
        )

        end_x, end_y = self._get_edge_end_point(
            sx1, sy1, sx2, sy2, r
        )

        edge_id = edge_data.get("id")
        selected = edge_id in self.selected_edges

        color = EDGE_SELECTED if selected else EDGE_COLOR
        width = 2 if selected else 1

        # IMPORTANT:
        # dashed lines with visible arrows
        line_id = self.canvas.create_line(
            start_x, start_y,
            cx, cy,
            end_x, end_y,
            smooth=True,
            splinesteps=18,
            fill=color,
            width=width,
            dash=(5, 4),
            arrow=tk.LAST,
            arrowshape=(
                max(8, int(12 * self.zoom)),
                max(10, int(12 * self.zoom)),
                max(3, int(4 * self.zoom))
            )
        )

        relation = edge_data.get("relation", "")

        # label
        if relation and self.zoom >= 0.60:

            tx = cx + px * 14 * direction
            ty = cy + py * 14 * direction

            text_id = self.canvas.create_text(
                tx,
                ty,
                text=relation,
                fill=TEXT_MAIN,
                font=(
                    "Consolas",
                    max(6, int(7 * self.zoom))
                )
            )

            bb = self.canvas.bbox(text_id)

            if bb:
                bg = self.canvas.create_rectangle(
                    bb[0] - 4,
                    bb[1] - 2,
                    bb[2] + 4,
                    bb[3] + 2,
                    fill=BG_DARK,
                    outline=LINE_GREEN_DARK,
                    width=1
                )
                self.canvas.tag_lower(bg, text_id)

        self.edge_items[line_id] = {
            "id": edge_id,
            "source": u,
            "target": v,
            "relation": relation,
            "canvas_id": line_id
        }

    # ======================================================
    # INTERACTION
    # ======================================================

    def on_press(self, event):

        if self.panning:
            return

        shift = (event.state & 0x0001) != 0

        halo = max(8, 14 * self.zoom)

        closest = self.canvas.find_closest(
            event.x,
            event.y,
            halo=halo
        )

        clicked_node = None
        clicked_edge = None

        for item in closest:

            if item in self.node_items:
                clicked_node = self.node_items[item]
                break

            if item in self.edge_items:
                clicked_edge = self.edge_items[item]
                break

        # NODE CLICK
        if clicked_node:

            if shift:
                self.selected_nodes.add(clicked_node)
            else:
                self.selected_nodes = {clicked_node}
                self.selected_edges.clear()
                self.is_editing = False

            sx, sy = self.screen_positions[clicked_node]

            self.dragging_node = clicked_node
            self.drag_offset_x = event.x - sx
            self.drag_offset_y = event.y - sy

            self.redraw()
            self.update_info_panel()
            return

        # EDGE CLICK
        if clicked_edge:

            eid = clicked_edge["id"]

            if shift:
                self.selected_edges.add(eid)
            else:
                self.selected_edges = {eid}
                self.selected_nodes.clear()
                self.is_editing = False

            self.redraw()
            self.update_info_panel()
            return

        # EMPTY CLICK
        if not shift:
            self.selected_nodes.clear()
            self.selected_edges.clear()
            self.is_editing = False

            self.redraw()
            self.update_info_panel()

# ===== CONTINUE NEXT MESSAGE =====

    # ------------------------------------------------------

    def on_drag(self, event):

        if not self.dragging_node:
            return

        if self.panning:
            return

        new_sx = event.x - self.drag_offset_x
        new_sy = event.y - self.drag_offset_y

        wx, wy = self.to_world(new_sx, new_sy)

        self.node_positions[self.dragging_node] = [wx, wy]

        self.redraw()

    # ------------------------------------------------------

    def on_release(self, event):
        self.dragging_node = None

    # ======================================================
    # PAN
    # ======================================================

    def on_pan_start(self, event):

        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y

        self.canvas.config(cursor="fleur")

    def on_pan(self, event):

        if not self.panning:
            return

        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y

        self.pan_start_x = event.x
        self.pan_start_y = event.y

        self.pan_offset[0] += dx / self.zoom
        self.pan_offset[1] += dy / self.zoom

        self.redraw()

    def on_pan_end(self, event):

        self.panning = False
        self.canvas.config(cursor="")

    # ======================================================
    # ZOOM
    # ======================================================

    def on_zoom(self, event):
        self._zoom_event(event.delta)

    def _zoom_event(self, delta):

        if delta == 0:
            return

        factor = 1.15 if delta > 0 else 0.87

        wx, wy = self.to_world(
            self.last_mouse_x,
            self.last_mouse_y
        )

        self.zoom *= factor

        self.zoom = max(0.15, min(5.0, self.zoom))

        self.pan_offset[0] = (
            self.last_mouse_x / self.zoom - wx
        )
        self.pan_offset[1] = (
            self.last_mouse_y / self.zoom - wy
        )

        self.redraw()

    # ======================================================
    # TRACK MOUSE
    # ======================================================

    def _track_mouse(self, event):
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    # ======================================================
    # RUN
    # ======================================================

    def run(self):
        self.root.mainloop()

# ==========================================================
# END
# ==========================================================