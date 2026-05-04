# All stuff about interactive graph visualization is here

#--------------------------
#---------imports----------
#--------------------------

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict
import networkx as nx
import math
import random
import sys

from nir.graph.graph_structures import Edge, Node, State

#-------------------------------
#---------theme config----------
#-------------------------------

BG_DARK = "#0a0a0a"
BG_NODE = "#151515" 
BG_PANEL = "#111111"

SECONDARY_COLOR = "#04681d"
PRIMARY_COLOR = "#0fc23c"
BUTTON_PRESSED_COLOR = "#077a24"
SELECT_COLOR = "#fffb20"

TEXT_COLOR = "#D1D1D1"
EDIT_MODE_COLOR = "#ffaa00"

RED_COLOR = "#ff4444"
RED_COLOR_PRESSED = "#c71717"
GREEN_COLOR = "#23bb49"
GREEN_COLOR_PRESSED = "#128b31"

TYPE_COLORS = {
    "character": "#ff9100",
    "group": "#ffb24d",
    "location": "#0077ff",
    "environment_element": "#00aeff",
    "item": "#db1b5e",
    "event": "#981aff",
    "default": "#949494"
}

TYPE_LABELS = {
    "character": "CHARACTER",
    "group": "GROUP OF CHARACTERS",
    "location": "LOCATION",
    "environment_element": "ENVIRONMENTAL ELEMENT",
    "item": "ITEM",
    "event": "EVENT",
}

TYPE_ZONES = {
    "character": {"x": -350, "y": -250},
    "group": {"x": -350, "y": -250},
    "location": {"x": 350, "y": -250},
    "event": {"x": 350, "y": 250},
    "environment_element": {"x": 350, "y": 250},
    "item": {"x": -350, "y": 250},
}

NODE_RADIUS = 30
EDGE_WIDTH_NORMAL = 1.5
EDGE_WIDTH_SELECTED = 3
CELL_SIZE = 130

ICON_EDIT = "📝"
ICON_DELETE = "🗑️"



#------------------------------------
#---------GraphWindow class----------
#------------------------------------

class GraphWindow:

    def __init__(self, graph_storage):
        self.storage = graph_storage
        self.graph = graph_storage.graph

        self.root = tk.Tk()
        self.root.title("Interactive graph visualization")
        self.root.geometry("1600x900")
        self.root.configure(bg=BG_DARK)

        self.zoom = 1.0
        self.pan_offset = [0.0, 0.0]
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

        self.setup_theme()
        self.build_ui()
        self._create_layout()
        self.redraw()
        self.update_info_panel()

    def setup_theme(self):
        style = ttk.Style()
        style.theme_use("clam")

        #basic button
        style.configure("TButton", 
                        background=BG_PANEL, 
                        foreground=PRIMARY_COLOR, 
                        bordercolor=BUTTON_PRESSED_COLOR,
                        focuscolor="",
                        relief="solid",
                        highlightthickness=0,
                        borderwidth=1,
                        font=("Consolas", 12),
                        padding=5)
        style.map("TButton",
                  foreground=[("active", PRIMARY_COLOR), ("pressed", BUTTON_PRESSED_COLOR)],
                  background=[("active", BG_PANEL), ("pressed", BG_PANEL)],
                  bordercolor=[("active", BUTTON_PRESSED_COLOR), ("pressed", BUTTON_PRESSED_COLOR)])
        
        #red button
        style.configure("Red.TButton", 
                        background=BG_PANEL, 
                        foreground=RED_COLOR, 
                        bordercolor=RED_COLOR_PRESSED, 
                        focuscolor="",
                        relief="solid",
                        highlightthickness=0,
                        borderwidth=1,
                        font=("Consolas", 12),
                        padding=5)
        style.map("Red.TButton",
                  foreground=[("active", RED_COLOR), ("pressed", RED_COLOR_PRESSED)],
                  background=[("active", BG_PANEL), ("pressed", BG_PANEL)],
                  bordercolor=[("active", RED_COLOR_PRESSED), ("pressed", RED_COLOR_PRESSED)])

        #green button
        style.configure("Green.TButton", 
                        background=BG_PANEL, 
                        foreground=GREEN_COLOR, 
                        bordercolor=GREEN_COLOR_PRESSED, 
                        focuscolor="",
                        relief="solid",
                        highlightthickness=0,
                        borderwidth=1,
                        font=("Consolas", 12),
                        padding=5)
        style.map("Green.TButton",
                  foreground=[("active", GREEN_COLOR), ("pressed", GREEN_COLOR_PRESSED)],
                  background=[("active", BG_PANEL), ("pressed", BG_PANEL)],
                  bordercolor=[("active", GREEN_COLOR_PRESSED), ("pressed", GREEN_COLOR_PRESSED)])
        
        style.configure("TFrame",  background=BG_PANEL)
        style.configure("TLabel", background=BG_PANEL, foreground=PRIMARY_COLOR, font=("Consolas", 11))
        style.configure("Title.TLabel", background=BG_PANEL, foreground=PRIMARY_COLOR, font=("Consolas", 10))
        style.configure("BasicText.TLabel", background=BG_PANEL, foreground=TEXT_COLOR, wraplength=320, font=("Consolas", 11))

        style.configure("TScrollbar", troughcolor=BG_DARK, background=BG_PANEL, arrowcolor=BG_PANEL)
        style.configure("Vertical.TScrollbar", 
                troughcolor=BG_DARK, 
                background=BG_PANEL, 
                arrowcolor=BG_PANEL,
                bordercolor=BG_PANEL,
                lightcolor=SECONDARY_COLOR,
                darkcolor=SECONDARY_COLOR,              
                width=6)
        style.map("Vertical.TScrollbar",
            background=[("active", BG_PANEL), ("pressed", BG_PANEL)],
            troughcolor=[("active", BG_DARK), ("pressed", BG_DARK)],
            arrowcolor=[("active", SECONDARY_COLOR), ("pressed", SECONDARY_COLOR)],
            lightcolor=[("active", SECONDARY_COLOR), ("pressed", SECONDARY_COLOR)],
            darkcolor=[("active", SECONDARY_COLOR), ("pressed", SECONDARY_COLOR)])
        
    def build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.columnconfigure(2, weight=0)
        self.root.rowconfigure(0, weight=1)

        left_container = ttk.Frame(self.root)
        left_container.grid(row=0, column=0, sticky="nsew")
        left_container.rowconfigure(2, weight=1)
        left_container.columnconfigure(0, weight=1)

        #header with title 
        header_frame = ttk.Frame(left_container, style="TFrame")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(header_frame, text="$ graph-editor --mode interactive", style="TLabel", anchor="w").pack(fill="x", padx=15)
        separator = tk.Frame(left_container, height=1, bg=SECONDARY_COLOR)
        separator.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        #canvas with graph
        self.canvas = tk.Canvas(left_container, bg=BG_DARK, highlightthickness=0, bd=0)
        self.canvas.grid(row=2, column=0, sticky="nsew", pady=(10, 0))

        self.sort_btn = ttk.Button(self.canvas, text="[↓ SORT_BY_TYPE]", command=self.toggle_sort, style="TButton")
        self.sort_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-15, y=15)

        #separator
        tk.Frame(self.root, width=1, bg=SECONDARY_COLOR).grid(row=0, column=1, sticky="ns")

        right_panel = tk.Frame(self.root, bg=BG_DARK, width=320)
        right_panel.grid(row=0, column=2, sticky="ns")
        right_panel.grid_propagate(False)
        self.build_right_panel(right_panel)

        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        self.canvas.bind("<Motion>", lambda e: self._track_mouse(e))
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_event(120))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_event(-120))
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        
        self.draw_static_legend()

            
    def build_right_panel(self, parent):
        info_header = tk.Frame(parent, bg=BG_DARK)
        info_header.pack(fill="x", padx=0, pady=(8, 5))
        ttk.Label(info_header, text="▶ INFO", style="TLabel", anchor="w").pack(anchor="w", padx=10)
        tk.Frame(info_header, height=1, bg=SECONDARY_COLOR).pack(fill="x", pady=(5, 0))

        self.info_container = tk.Frame(parent, bg=BG_DARK)
        self.info_container.pack(fill="both", expand=True, padx=15, pady=(5, 10))
        self.info_container.pack_propagate(False)
        
        self.info_frame = tk.Frame(self.info_container, bg=BG_PANEL, bd=2, relief="solid", 
                                highlightbackground=SECONDARY_COLOR, highlightcolor=SECONDARY_COLOR, 
                                highlightthickness=1, width=288)
        self.info_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        self.info_frame.grid_rowconfigure(0, weight=1)
        self.info_frame.grid_columnconfigure(0, weight=1)
        self.info_frame.grid_columnconfigure(1, weight=0)

        self.info_canvas = tk.Canvas(self.info_frame, bg=BG_PANEL, highlightthickness=0)
        self.info_scrollbar = ttk.Scrollbar(self.info_frame, orient="vertical", command=self.info_canvas.yview)
        self.info_canvas.configure(yscrollcommand=self.info_scrollbar.set)
        self.info_canvas.grid(row=0, column=0, sticky="nsew")

        self.info_scrollbar.grid(row=0, column=1, sticky="ns")
        self.info_scrollbar.grid_remove()

        self.info_content = ttk.Frame(self.info_canvas, style="TFrame")
        self.canvas_window = self.info_canvas.create_window((0, 0), window=self.info_content, anchor="nw") 

        legend_frame = ttk.Frame(parent, style="TFrame")
        legend_frame.pack(fill="x", side="bottom", pady=5)
        tk.Frame(legend_frame, height=1, bg=SECONDARY_COLOR).pack(fill="x", pady=(5, 10))
        
        ttk.Label(legend_frame, text="╔═ LEGEND ═╗", style="TLabel").pack(anchor="sw", padx=15)
        self.legend_canvas = tk.Canvas(legend_frame, bg=BG_PANEL, height=160, highlightthickness=0)
        self.legend_canvas.pack(fill="both", padx=15, pady=(0, 5))

        self.update_scrollbar()
        self.info_canvas.bind("<Configure>", lambda e: self.info_canvas.itemconfig(self.canvas_window, width=e.width))
        self.info_content.bind("<Configure>", lambda e: self.update_scrollbar())
        self.setup_info_panel_scroll()

    #----------------------------------------
    #---------Scrolling information----------
    #----------------------------------------

    def update_scrollbar(self):
        self.info_canvas.update_idletasks()
        self.info_canvas.configure(scrollregion=self.info_canvas.bbox("all"))
        bbox = self.info_canvas.bbox("all")
        if bbox is None:
            self.info_scrollbar.grid_remove()
            return
        canvas_height = self.info_canvas.winfo_height()
        if bbox[3] > canvas_height:
            self.info_scrollbar.grid()
        else:
            self.info_scrollbar.grid_remove()

    def setup_info_panel_scroll(self):
        def on_mousewheel(event):
            if hasattr(self, 'info_scrollbar') and self.info_scrollbar.winfo_viewable():
                self.info_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                return "break"
        def on_mousewheel_linux(event):
            if hasattr(self, 'info_scrollbar') and self.info_scrollbar.winfo_viewable():
                if event.num == 4:
                    self.info_canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.info_canvas.yview_scroll(1, "units")
                return "break"
        def bind_recursive(widget):
            if sys.platform in ('darwin', 'win32'):
                widget.bind("<MouseWheel>", on_mousewheel, add="+")
            else:
                widget.bind("<Button-4>", on_mousewheel_linux, add="+")
                widget.bind("<Button-5>", on_mousewheel_linux, add="+")
            for child in widget.winfo_children():
                bind_recursive(child)
        if sys.platform in ('darwin', 'win32'):
            self.info_canvas.bind("<MouseWheel>", on_mousewheel)
            self.info_content.bind("<MouseWheel>", on_mousewheel)
        else:
            self.info_canvas.bind("<Button-4>", on_mousewheel_linux)
            self.info_canvas.bind("<Button-5>", on_mousewheel_linux)
            self.info_content.bind("<Button-4>", on_mousewheel_linux)
            self.info_content.bind("<Button-5>", on_mousewheel_linux)
        bind_recursive(self.info_content)

    def bind_scroll_to_widgets(self):
        def on_mousewheel(event):
            if hasattr(self, 'info_scrollbar') and self.info_scrollbar.winfo_viewable():
                self.info_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                return "break"
        def on_mousewheel_linux(event):
            if hasattr(self, 'info_scrollbar') and self.info_scrollbar.winfo_viewable():
                if event.num == 4:
                    self.info_canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.info_canvas.yview_scroll(1, "units")
                return "break"
        def bind_recursive(widget):
            if sys.platform in ('darwin', 'win32'):
                widget.bind("<MouseWheel>", on_mousewheel, add="+")
            else:
                widget.bind("<Button-4>", on_mousewheel_linux, add="+")
                widget.bind("<Button-5>", on_mousewheel_linux, add="+")
            for child in widget.winfo_children():
                bind_recursive(child)
        bind_recursive(self.info_content)



    #-----------------------------------------------
    #---------Updating and redrawing logic----------
    #-----------------------------------------------

    def add_bottom_hint(self):
        zoom_text = f"ZOOM: {int(self.zoom * 100)}%"
        controls_text = "scroll=zoom | middledrag=pan | shift+click=multi"
        
        if hasattr(self, 'hint_frame') and self.hint_frame:
            for child in self.hint_frame.winfo_children():
                if isinstance(child, tk.Label):
                    if child._text_id == "zoom":
                        child.config(text=zoom_text, foreground=PRIMARY_COLOR)
                    elif child._text_id == "controls":
                        child.config(text=controls_text, foreground=SECONDARY_COLOR)
        else:
            self.hint_frame = tk.Frame(self.canvas, bg=BG_PANEL, bd=2, relief="solid", 
                                    highlightbackground=SECONDARY_COLOR, 
                                    highlightcolor=SECONDARY_COLOR, 
                                    highlightthickness=1)
            self.hint_frame.place(relx=0, rely=1, anchor="sw", x=15, y=-15)
            zoom_label = tk.Label(self.hint_frame, text=zoom_text, fg=PRIMARY_COLOR, bg=BG_PANEL, 
                                font=("Consolas", 9), justify="left")
            zoom_label._text_id = "zoom"
            zoom_label.pack(anchor="w", padx=12, pady=(8, 0))
    
            separator = tk.Frame(self.hint_frame, height=1, bg=SECONDARY_COLOR)
            separator.pack(fill="x", padx=12, pady=(5, 5))
            controls_label = tk.Label(self.hint_frame, text=controls_text, fg=SECONDARY_COLOR, bg=BG_PANEL, 
                                    font=("Consolas", 9), justify="left")
            controls_label._text_id = "controls"
            controls_label.pack(anchor="w", padx=12, pady=(0, 8))
            
    def update_info_panel(self):
        for widget in self.info_content.winfo_children():
            widget.destroy()
        if self.is_editing and len(self.selected_nodes) == 1:
            self.draw_node_edit_mode()
        elif self.is_editing and len(self.selected_edges) == 1:
            self.draw_edge_edit_mode()
        elif len(self.selected_nodes) > 1:
            self.draw_multi_select()
        elif len(self.selected_nodes) == 1:
            self.draw_node_info()
        elif len(self.selected_edges) == 1:
            self.draw_edge_info()
        else:
            self.draw_empty_state()
        self.bind_scroll_to_widgets()

        if hasattr(self, 'info_canvas'):
            self.info_canvas.update_idletasks()
            self.info_canvas.configure(scrollregion=self.info_canvas.bbox("all"))
            
            bbox = self.info_canvas.bbox("all")
            canvas_height = self.info_canvas.winfo_height()
            if bbox and bbox[3] > canvas_height:
                self.info_scrollbar.grid()
            else:
                self.info_scrollbar.grid_remove()
        self.update_scrollbar()

    def draw_empty_state(self):
        self.info_content.grid_rowconfigure(0, weight=1)
        self.info_content.grid_columnconfigure(0, weight=1)
        center_frame = ttk.Frame(self.info_content)
        center_frame.grid(row=0, column=0)
        text = "┌──────────────┐\n\n"
        text += "│ NO SELECTION │\n\n"
        text += "└──────────────┘\n\n"
        tk.Label(center_frame, text=text, fg=SECONDARY_COLOR, bg=BG_PANEL, font=("Consolas", 11), justify="center").pack(pady=(30, 5))
        tk.Label(center_frame, text="Click node/edge to view", fg=SECONDARY_COLOR, font=("Consolas", 10), bg=BG_PANEL).pack()


    #------------------------------------------
    #---------Node info drawing logic----------
    #------------------------------------------
    
    def create_collapsible_section(self, parent, title, content_generator, is_open=True):
        section = ttk.Frame(parent, style="TFrame")
        section.pack(fill="x", pady=(5, 0))
    
        header = ttk.Frame(section, style="TFrame")
        header.pack(fill="x")

        is_expanded = tk.BooleanVar(value=is_open)
        arrow_symbol = "▼" if is_open else "▶"
        arrow_label = ttk.Label(header, text=f"{arrow_symbol} {title}", style="Title.TLabel", cursor="hand2")
        arrow_label.pack(side="left")
        
        def toggle():
            current = is_expanded.get()
            is_expanded.set(not current)
            arrow_label.config(text=f"{'▼' if not current else '▶'} {title}")
            if content_frame.winfo_ismapped():
                content_frame.pack_forget()
            else:
                content_frame.pack(fill="x", pady=(2, 0))
            self.root.after(100, self.update_scrollbar)

        header.bind("<Button-1>", lambda e: toggle(), add="+")
        arrow_label.bind("<Button-1>", lambda e: toggle(), add="+")
        content_frame = ttk.Frame(section, style="TFrame")
        if is_open:
            content_frame.pack(fill="x", pady=(2, 0))
        content_generator(content_frame)
        return section, content_frame, is_expanded

    def draw_state_card(self, parent, state: State, index: int):
        card = ttk.Frame(parent, style="TFrame")
        card.pack(fill="x", pady=(3, 0), padx=(10, 10))
        card_inner = tk.Frame(card, bg=BG_NODE, bd=1, relief="solid", highlightbackground=SECONDARY_COLOR, highlightthickness=1)
        card_inner.pack(fill="x", pady=2, padx=2)

        header = tk.Frame(card_inner, bg=BG_NODE)
        header.pack(fill="x", padx=8, pady=(2, 2))
        tk.Label(header, text=f"State #{index+1} [{state.sid}]", fg=SECONDARY_COLOR, bg=BG_NODE, font=("Consolas", 10), wraplength=300, justify="left").pack(anchor="w")

        if state.time_start_event or state.time_end_event:
            time_text = f"{state.time_start_event or '?'} – {state.time_end_event or '?'}"
            tk.Label(header, text=time_text, fg=TEXT_COLOR, bg=BG_NODE, font=("Consolas", 10), justify="left").pack(anchor="w")

        if state.current_description:
            tk.Label(card_inner, text=state.current_description, fg=TEXT_COLOR, bg=BG_NODE,
                    font=("Consolas", 9), wraplength=300, justify="left").pack(anchor="w", padx=8, pady=(0, 4))
        if state.current_attributes:
            attrs_grid_frame = tk.Frame(card_inner, bg=BG_NODE)
            attrs_grid_frame.pack(fill="x", padx=8, pady=(0, 6))
            attrs_grid_frame.columnconfigure(0, weight=0)
            attrs_grid_frame.columnconfigure(1, weight=1)
            value_labels = []

            for i, (key, value) in enumerate(state.current_attributes.items()):
                val_str = str(value)
                if len(val_str) > 500: 
                    val_str = val_str[:500] + "..."
                tk.Label(attrs_grid_frame, text=f"{key}:", 
                        fg=SECONDARY_COLOR, bg=BG_NODE, 
                        font=("Consolas", 9, "bold"), 
                        anchor="w").grid(row=i, column=0, sticky="e", padx=(0, 10), pady=1)

                val_lbl = tk.Label(attrs_grid_frame, text=val_str, 
                                fg=TEXT_COLOR, bg=BG_NODE, 
                                font=("Consolas", 9), 
                                anchor="nw", justify="left",
                                wraplength=280) 
                val_lbl.grid(row=i, column=1, sticky="ew", pady=1)      
                value_labels.append(val_lbl)
            def on_grid_resize(event):
                available_width = event.width - 110 
                if available_width < 50: 
                    available_width = 50 
                for lbl in value_labels:
                    lbl.config(wraplength=available_width)
            attrs_grid_frame.bind("<Configure>", on_grid_resize)

    def draw_attributes_block(self, parent, attributes: Dict[str, Any], title="ATTRIBUTES"):
        if not attributes:
            return
        ttk.Label(parent, text=f"{title}:", style="Title.TLabel").pack(anchor="w", pady=(8, 2))
        attrs_grid = ttk.Frame(parent, style="TFrame")
        attrs_grid.pack(fill="x")
        for i, (key, value) in enumerate(attributes.items()):
            ttk.Label(attrs_grid, text=f"{key}:", style="Title.TLabel", foreground=SECONDARY_COLOR, anchor="e").grid(row=i, column=0, sticky="e", padx=(0, 5))
            val_lbl = ttk.Label(attrs_grid, text=f"{str(value) or '?'}", style="BasicText.TLabel", justify="left")
            val_lbl.grid(row=i, column=1, sticky="w")
        
    def draw_node_info(self):
        node_id = list(self.selected_nodes)[0]
        node = self.storage.get_node_by_id(node_id)
        node_type = node.type
        color = TYPE_COLORS.get(node_type, TYPE_COLORS["default"])
        header_frame = tk.Frame(self.info_content, bg=BG_PANEL)
        header_frame.pack(fill="x", pady=5)
        
        ttk.Label(header_frame, text=f" ● [NODE] {node_type}", style="TLabel", foreground=color).pack(side="left")

        delete_btn = ttk.Button(header_frame, text=f"{ICON_DELETE}", command=lambda: self._delete_node(node_id), padding=(9, 0, 0, 5), style="Red.TButton", width=3)
        delete_btn.pack(side="right", padx=5)
        edit_btn = ttk.Button(header_frame, text=f"{ICON_EDIT}", command=lambda: self._start_editing(node_id), padding=(2, 0, 2, 5), style="Green.TButton", width=3)
        edit_btn.pack(side="right", padx=5)
        #separator
        tk.Frame(self.info_content, height=1, bg=SECONDARY_COLOR).pack(fill="x", pady=(5, 10), padx=10)
        
        ttk.Label(self.info_content, text="NAME", style="Title.TLabel").pack(anchor="w", pady=(10, 2))
        ttk.Label(self.info_content, text=f"{node.name}", style="BasicText.TLabel").pack(anchor="w", pady=(0, 10))

        ttk.Label(self.info_content, text="BASE DESCRIPTION", style="Title.TLabel").pack(anchor="w", pady=(10, 2))
        ttk.Label(self.info_content, text=f"{node.base_description}", style="BasicText.TLabel").pack(anchor="w", pady=(0, 10))

        #separator
        if node.base_attributes or node.states:
            tk.Frame(self.info_content, height=1, bg=SECONDARY_COLOR).pack(fill="x", pady=(5, 10), padx=10)

        if node.base_attributes:
            self.draw_attributes_block(self.info_content, node.base_attributes, "BASE ATTRIBUTES")
        
        if node.states:
            def generate_states_content(parent):
                for idx, state in enumerate(node.states):
                    self.draw_state_card(parent, state, idx)
            self.create_collapsible_section(
                self.info_content, 
                f"STATES ({len(node.states)})", 
                generate_states_content,
                is_open=True
            )

    def create_edit_label(self, parent, text):
        ttk.Label(parent, text=text, style="Title.TLabel",).pack(anchor="w", pady=(8, 2))

    def create_entry(self, parent, text_var):
        entry = tk.Entry(parent, textvariable=text_var, bg=BG_DARK, fg=EDIT_MODE_COLOR, 
                         insertbackground=EDIT_MODE_COLOR, font=("Consolas", 10), 
                         bd=2, relief="solid")
        entry.config(highlightbackground=EDIT_MODE_COLOR, highlightcolor=EDIT_MODE_COLOR, highlightthickness=1)
        return entry

    def create_text_widget(self, parent, initial_text):
        txt = tk.Text(parent, bg=BG_DARK, fg=EDIT_MODE_COLOR, insertbackground=EDIT_MODE_COLOR, 
                      font=("Consolas", 9), bd=2, relief="solid", wrap="word", 
                      height=max(3, initial_text.count('\n') + 1))
        txt.insert("1.0", initial_text)
        txt.config(highlightbackground=EDIT_MODE_COLOR, highlightcolor=EDIT_MODE_COLOR, highlightthickness=1)
        return txt
    
    def calculate_text_height(self, text):
        if not text:
            return 4
        chars_per_line = 35 
        lines = (len(text) / chars_per_line) + text.count('\n')
        return max(4, int(lines) + 1)

    def create_text_widget(self, parent, initial_text):
        txt = tk.Text(parent, bg=BG_DARK, fg=EDIT_MODE_COLOR, insertbackground=EDIT_MODE_COLOR, 
                      font=("Consolas", 9), bd=2, relief="solid", wrap="word", 
                      height=self.calculate_text_height(initial_text))
        txt.insert("1.0", initial_text)
        txt.config(highlightbackground=EDIT_MODE_COLOR, highlightcolor=EDIT_MODE_COLOR, highlightthickness=1)
        return txt

    def draw_node_edit_mode(self):
        node_id = list(self.selected_nodes)[0]
        node = self.storage.get_node_by_id(node_id)

        for widget in self.info_content.winfo_children():
            widget.destroy()
        self.edit_widgets = {}

        main_container = tk.Frame(self.info_content, bg=BG_PANEL)
        main_container.pack(fill="both", expand=True)

        content_area = tk.Frame(main_container, bg=BG_PANEL)
        content_area.pack(fill="both", expand=True)

        button_area = tk.Frame(main_container, bg=BG_PANEL)
        button_area.pack(fill="x", pady=(10, 5))

        header_frame = tk.Frame(content_area, bg=BG_PANEL)
        header_frame.pack(fill="x", pady=5)
        tk.Label(header_frame, text="EDIT NODE MODE", fg=EDIT_MODE_COLOR, bg=BG_PANEL, font=("Consolas", 11)).pack(side="left")
        
        tk.Frame(content_area, height=1, bg=SECONDARY_COLOR).pack(fill="x", pady=(5, 10), padx=10)
        
        self.create_edit_label(content_area, "NAME")
        self.edit_widgets["name"] = tk.StringVar(value=node.name)
        self.create_entry(content_area, self.edit_widgets["name"]).pack(anchor="w", pady=(0, 10), fill="x")
        
        self.create_edit_label(content_area, "BASE DESCRIPTION")
        self.edit_widgets["base_description"] = self.create_text_widget(content_area, node.base_description or "")
        self.edit_widgets["base_description"].pack(anchor="w", pady=(0, 15), fill="x")
        
        self.edit_widgets["base_attributes"] = {}
        if node.base_attributes:
            self.create_edit_label(content_area, "BASE ATTRIBUTES")
            attrs_frame = tk.Frame(content_area, bg=BG_PANEL)
            attrs_frame.pack(fill="x", pady=(0, 10))
            
            for key, value in node.base_attributes.items():
                row = tk.Frame(attrs_frame, bg=BG_PANEL)
                row.pack(fill="x", pady=2)
                tk.Label(row, text=f"{key}:", fg=SECONDARY_COLOR, bg=BG_PANEL, 
                        font=("Consolas", 10, "bold"), anchor="e", width=14).pack(side="left", padx=(0, 5))
                
                var = tk.StringVar(value=str(value))
                self.edit_widgets["base_attributes"][key] = var
                self.create_entry(row, var).pack(side="left", fill="x", expand=True)
        
        self.edit_widgets["states"] = []
        if node.states:
            self.create_edit_label(content_area, f"STATES ({len(node.states)})")
            
            for idx, state in enumerate(node.states):
                def build_state_ui(parent, s=state, i=idx):
                    state_data = {"sid": s.sid, "attributes": {}}

                    state_header = tk.Frame(parent, bg=BG_NODE)
                    state_header.pack(fill="x", padx=8, pady=(5, 2))
                    tk.Label(state_header, text=f"State #{i+1} [{s.sid}]", 
                            fg=SECONDARY_COLOR, bg=BG_NODE, 
                            font=("Consolas", 10, "bold")).pack(anchor="w")

                    time_frame = tk.Frame(parent, bg=BG_NODE)
                    time_frame.pack(fill="x", padx=8, pady=2)
                    tk.Label(time_frame, text="TIME START:", fg=TEXT_COLOR, bg=BG_NODE, 
                            font=("Consolas", 9)).pack(anchor="w", pady=(0, 2))
                    var_start = tk.StringVar(value=s.time_start_event or "")
                    state_data["time_start"] = var_start
                    self.create_entry(time_frame, var_start).pack(fill="x", expand=True, pady=(0, 5))

                    tk.Label(time_frame, text="TIME END:", fg=TEXT_COLOR, bg=BG_NODE, 
                            font=("Consolas", 9)).pack(anchor="w", pady=(2, 2))
                    var_end = tk.StringVar(value=s.time_end_event or "")
                    state_data["time_end"] = var_end
                    self.create_entry(time_frame, var_end).pack(fill="x", expand=True, pady=(0, 5))

                    self.create_edit_label(parent, "DESCRIPTION")
                    state_data["description"] = self.create_text_widget(parent, s.current_description or "")
                    state_data["description"].pack(anchor="w", padx=8, pady=(0, 6), fill="x")

                    if s.current_attributes:
                        self.create_edit_label(parent, "ATTRIBUTES")
                        attr_frame = tk.Frame(parent, bg=BG_NODE)
                        attr_frame.pack(fill="x", padx=8, pady=(0, 6))
                        
                        for key, value in s.current_attributes.items():
                            row = tk.Frame(attr_frame, bg=BG_NODE)
                            row.pack(fill="x", pady=1)
                            tk.Label(row, text=f"{key}:", fg=SECONDARY_COLOR, bg=BG_NODE, 
                                    font=("Consolas", 9, "bold"), anchor="e", width=10).pack(side="left", padx=(0, 5))
                            var = tk.StringVar(value=str(value))
                            state_data["attributes"][key] = var
                            self.create_entry(row, var).pack(side="left", fill="x", expand=True)
                    
                    self.edit_widgets["states"].append(state_data)
                
                self.create_collapsible_section(
                    content_area, 
                    f"State #{idx+1} [{state.sid}]", 
                    build_state_ui, 
                    is_open=True
                )

        btn_container = tk.Frame(button_area, bg=BG_PANEL)
        btn_container.pack(pady=10)
        ttk.Button(btn_container, text="[ SAVE ]", command=lambda: self.save_edit_node(node_id), style="Green.TButton").pack(side="left", padx=5)
        ttk.Button(btn_container, text="[ CANCEL ]", command=self.cancel_edit, style="Red.TButton").pack(side="left", padx=5)
        self.root.after(100, self.update_scrollbar)

    def save_edit_node(self, node_id):
        original_node = self.storage.get_node_by_id(node_id)
        updated_node = Node(
            id=node_id,
            name=self.edit_widgets["name"].get().strip(),
            type=original_node.type,
            base_description=self.edit_widgets["base_description"].get("1.0", "end-1c").strip(),
            base_attributes={},
            states=[],
            chunk_id=original_node.chunk_id
        )

        base_attrs = self.edit_widgets.get("base_attributes", {})
        for key, var in base_attrs.items():
            updated_node.base_attributes[key] = self._parse_value(var.get().strip())

        saved_states = self.edit_widgets.get("states", [])
        for i, state_data in enumerate(saved_states):
            if i < len(original_node.states):
                sid = original_node.states[i].sid
            else:
                sid = f"state_{i}_{node_id}"
 
            updated_state = State(
                sid=sid,
                current_description=state_data["description"].get("1.0", "end-1c").strip(),
                current_attributes={},
                time_start_event=self._parse_value(state_data["time_start"].get().strip()) or None,
                time_end_event=self._parse_value(state_data["time_end"].get().strip()) or None
            )

            for key, var in state_data["attributes"].items():
                updated_state.current_attributes[key] = self._parse_value(var.get().strip())
            updated_node.states.append(updated_state)

        self.storage.update_node_full(node_id, updated_node)
        self.graph = self.storage.graph
        #saving logic
        self.is_editing = False
        self.update_info_panel()
        self.redraw()

    def draw_edge_info(self):
        if not self.selected_edges:
            return
        edge_id = list(self.selected_edges)[0]
        edge_data = self.storage.get_edge_by_id(edge_id)
                    
        header_frame = tk.Frame(self.info_content, bg=BG_PANEL)
        header_frame.pack(fill="x", pady=5)
        ttk.Label(header_frame, text=f" 🔗 [EDGE]", style="TLabel").pack(side="left")
        
        delete_btn = ttk.Button(header_frame, text=f"{ICON_DELETE}", command=lambda: self._delete_edge(edge_id), padding=(9, 0, 0, 5), style="Red.TButton", width=3)
        delete_btn.pack(side="right", padx=5)
        edit_btn = ttk.Button(header_frame, text=f"{ICON_EDIT}", command=lambda: self._start_editing(edge_id), padding=(2, 0, 2, 5), style="Green.TButton", width=3)
        edit_btn.pack(side="right", padx=5)
        #separator
        tk.Frame(self.info_content, height=1, bg=SECONDARY_COLOR).pack(fill="x", pady=(5, 10), padx=10)

        ttk.Label(self.info_content, text="RELATION", style="Title.TLabel").pack(anchor="w", pady=(10, 2))
        ttk.Label(self.info_content, text=edge_data.relation, style="BasicText.TLabel").pack(anchor="w", pady=(0, 10))
        
        if edge_data.description:
            ttk.Label(self.info_content, text="DESCRIPTION", style="Title.TLabel").pack(anchor="w", pady=(10, 2))
            ttk.Label(self.info_content, text=edge_data.description, style="BasicText.TLabel").pack(anchor="w", pady=(0, 10))

        if edge_data.time_start_event or edge_data.time_end_event:
            time_text = f"{edge_data.time_start_event or '?'} – {edge_data.time_end_event or '?'}"
            ttk.Label(self.info_content, text="TIME RANGE", style="Title.TLabel").pack(anchor="w", pady=(10, 2))
            ttk.Label(self.info_content, text=time_text, style="BasicText.TLabel").pack(anchor="w", pady=(0, 10))

        ttk.Label(self.info_content, text="WEIGHT", style="Title.TLabel").pack(anchor="w", pady=(10, 2))
        ttk.Label(self.info_content, text=str(edge_data.weight), style="BasicText.TLabel").pack(anchor="w", pady=(0, 10))

    def draw_edge_edit_mode(self):
        edge_id = list(self.selected_edges)[0]
        edge_data = self.storage.get_edge_by_id(edge_id)
        if not edge_data: 
            return

        for widget in self.info_content.winfo_children():
            widget.destroy()
        self.edit_widgets = {"edge": {}}

        main_container = tk.Frame(self.info_content, bg=BG_PANEL)
        main_container.pack(fill="both", expand=True)

        content_area = tk.Frame(main_container, bg=BG_PANEL)
        content_area.pack(fill="both", expand=True)

        button_area = tk.Frame(main_container, bg=BG_PANEL)
        button_area.pack(fill="x", pady=(10, 5))
        
        header_frame = tk.Frame(content_area, bg=BG_PANEL)
        header_frame.pack(fill="x", pady=5)
        tk.Label(header_frame, text="EDIT EDGE MODE", fg=EDIT_MODE_COLOR, bg=BG_PANEL, 
                font=("Consolas", 11)).pack(side="left")
        tk.Frame(header_frame, height=1, bg=SECONDARY_COLOR).pack(fill="x", pady=(5, 10), padx=10)

        self.create_edit_label(content_area, "RELATION")
        self.edit_widgets["edge"]["relation"] = tk.StringVar(value=edge_data.relation)
        self.create_entry(content_area, self.edit_widgets["edge"]["relation"]).pack(anchor="w", pady=(0, 10), fill="x")

        self.create_edit_label(content_area, "DESCRIPTION")
        self.edit_widgets["edge"]["description"] = self.create_text_widget(content_area, edge_data.description)
        self.edit_widgets["edge"]["description"].pack(anchor="w", pady=(0, 15), fill="x")

        self.create_edit_label(content_area, "WEIGHT")
        self.edit_widgets["edge"]["weight"] = tk.StringVar(value=str(edge_data.weight))
        self.create_entry(content_area, self.edit_widgets["edge"]["weight"]).pack(anchor="w", pady=(0, 10), fill="x")

        time_frame = tk.Frame(content_area, bg=BG_PANEL)
        time_frame.pack(fill="x", pady=(0, 10))
        
        self.create_edit_label(time_frame, "TIME START:")
        self.edit_widgets["edge"]["time_start"] = tk.StringVar(value=edge_data.time_start_event or "")
        self.create_entry(time_frame, self.edit_widgets["edge"]["time_start"]).pack(fill="x", expand=True, padx=(5, 0))
        
        self.create_edit_label(time_frame, "TIME END:")
        self.edit_widgets["edge"]["time_end"] = tk.StringVar(value=edge_data.time_end_event or "")
        self.create_entry(time_frame, self.edit_widgets["edge"]["time_end"]).pack(fill="x", expand=True, padx=(5, 0))

        btn_container = tk.Frame(button_area, bg=BG_PANEL)
        btn_container.pack(pady=10)
        ttk.Button(btn_container, text="[ SAVE ]", command=lambda: self.save_edit_edge(edge_id), style="Green.TButton").pack(side="left", padx=5)
        ttk.Button(btn_container, text="[ CANCEL ]", command=self.cancel_edit, style="Red.TButton").pack(side="left", padx=5)
    
    def save_edit_edge(self, edge_id):
        original_edge = self.storage.get_edge_by_id(edge_id)
        
        if not original_edge:
            print(f"Edge {edge_id} not found!")
            return
        
        updated_edge = Edge(
            id=edge_id,
            source=original_edge.source,
            target=original_edge.target,
            relation=self.edit_widgets["edge"]["relation"].get().strip(),
            description=self.edit_widgets["edge"]["description"].get("1.0", "end-1c").strip(),
            weight=self._parse_value(self.edit_widgets["edge"]["weight"].get().strip()) or 1.0,
            time_start_event=str(self._parse_value(self.edit_widgets["edge"]["time_start"].get().strip())) or None,
            time_end_event=str(self._parse_value(self.edit_widgets["edge"]["time_end"].get().strip())) or None,
            chunk_id=original_edge.chunk_id
        )
        
        self.storage.update_edge_full(edge_id, updated_edge)
        self.graph = self.storage.graph
        self.is_editing = False
        self.update_info_panel()
        self.redraw()

    def draw_multi_select(self):
        main_container = tk.Frame(self.info_content, bg=BG_PANEL)
        main_container.pack(fill="both", expand=True)
        
        # Заголовок
        header_frame = tk.Frame(main_container, bg=BG_PANEL)
        header_frame.pack(fill="x", pady=(0, 5))
        tk.Label(header_frame, text=f"[ MULTI_SELECT ] {len(self.selected_nodes)} nodes", 
                fg=SELECT_COLOR, bg=BG_PANEL, font=("Consolas", 10, "bold")).pack(anchor="w")
        
        # Контейнер для списка (который будет растягиваться)
        list_container = tk.Frame(main_container, bg=BG_PANEL)
        list_container.pack(fill="both", expand=True, pady=5)
        
        # Canvas с фиксированной шириной и прокруткой
        nodes_canvas = tk.Canvas(list_container, bg=BG_PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=nodes_canvas.yview)
        nodes_canvas.configure(yscrollcommand=scrollbar.set)
        
        nodes_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Внутренний фрейм для узлов с фиксированной шириной
        nodes_frame = tk.Frame(nodes_canvas, bg=BG_PANEL)
        canvas_window = nodes_canvas.create_window((0, 0), window=nodes_frame, anchor="nw")
        
        # Функция обновления прокрутки
        def update_scroll_region(event=None):
            nodes_canvas.configure(scrollregion=nodes_canvas.bbox("all"))
        
        # Функция для установки ширины фрейма
        def set_frame_width(event):
            canvas_width = event.width
            nodes_canvas.itemconfig(canvas_window, width=canvas_width)
            # Принудительно обновляем прокрутку после изменения ширины
            nodes_canvas.update_idletasks()
            update_scroll_region()
        
        nodes_canvas.bind("<Configure>", set_frame_width)
        nodes_frame.bind("<Configure>", update_scroll_region)
        
        # Отображаем каждый узел в отдельной карточке с рамкой
        for nid in self.selected_nodes:
            node = self.storage.get_node_by_id(nid)
            node_type = node.type
            color = TYPE_COLORS.get(node_type, TYPE_COLORS["default"])
            
            # Карточка узла с фиксированной шириной (за счёт fill="x" в родителе)
            card = tk.Frame(nodes_frame, bg=BG_NODE, bd=2, relief="solid", 
                        highlightbackground=color, highlightcolor=color, 
                        highlightthickness=1)
            card.pack(fill="x", pady=3)
            
            # Цветной индикатор
            indicator = tk.Frame(card, bg=color, width=4)
            indicator.pack(side="left", fill="y", padx=(0, 8))
            indicator.pack_propagate(False)
            
            # Имя узла (растягивается)
            tk.Label(card, text=node.name, fg=color, bg=BG_NODE, 
                    font=("Consolas", 10, "bold"), anchor="w").pack(side="left", fill="x", expand=True, padx=5, pady=5)
            
            # Тип узла
            tk.Label(card, text=node_type.upper(), fg=TEXT_COLOR, bg=BG_NODE, 
                    font=("Consolas", 8)).pack(side="right", padx=5, pady=5)
        
        # Кнопки внизу (приклеены к низу)
        buttons_frame = tk.Frame(main_container, bg=BG_PANEL)
        buttons_frame.pack(fill="x", pady=(5, 0))
        
        # Создаём внутренний фрейм для кнопок с отступами
        buttons_inner = tk.Frame(buttons_frame, bg=BG_PANEL)
        buttons_inner.pack(fill="x", pady=10)
        
        merge_btn = ttk.Button(buttons_inner, text="MERGE NODES", 
                            command=self._merge_nodes, style="Green.TButton")
        merge_btn.pack(side="left", padx=5, fill="x", expand=True)
        
        delete_btn = ttk.Button(buttons_inner, text="DELETE SELECTED", 
                            command=lambda: self._delete_selected_elements(), style="Red.TButton")
        delete_btn.pack(side="left", padx=5, fill="x", expand=True)
        
        # Принудительно обновляем прокрутку после отрисовки
        self.root.after(100, update_scroll_region)

    def _delete_selected_elements(self):
        """Удаляет выбранные узлы и рёбра"""
        
        # Удаляем выбранные узлы
        for node_id in list(self.selected_nodes):
            if node_id in self.graph.nodes:
                self.graph.remove_node(node_id)
                # Также удаляем из storage, если есть такой метод
                if hasattr(self.storage, 'remove_node'):
                    self.storage.remove_node(node_id)
        
        # Удаляем выбранные рёбра
        for edge_id in list(self.selected_edges):
            # Находим ребро в графе и удаляем
            for u, v, key, attrs in self.graph.edges(keys=True, data=True):
                if "data" in attrs and attrs["data"].get("id") == edge_id:
                    self.graph.remove_edge(u, v, key)
                    # Также удаляем из storage, если есть такой метод
                    if hasattr(self.storage, 'remove_edge'):
                        self.storage.remove_edge(edge_id)
                    break
        
        # Очищаем выбор
        self.selected_nodes.clear()
        self.selected_edges.clear()
        
        # Обновляем граф из storage
        self.graph = self.storage.graph
        
        # Перерисовываем всё
        self.redraw()
        self.update_info_panel()
        self._add_bottom_hint()

    def _start_editing(self, node_id):
        self.is_editing = True
        self.update_info_panel()

    def cancel_edit(self):
        self.is_editing = False
        self.update_info_panel()

    @staticmethod
    def _parse_value(val: str):
        if not val:
            return None
        if val.lower() in ("true", "false"):
            return val.lower() == "true"
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

    def _delete_node(self, node_id):
        self.graph.remove_node(node_id)
        self.selected_nodes.clear()
        self.redraw()
        self.update_info_panel()

    def _delete_edge(self, edge_id):
        pass

    def _merge_nodes(self):
        """Объединяет выбранные узлы в один"""
        if len(self.selected_nodes) < 2:
            print("Need at least 2 nodes to merge")
            return
        
        # TODO: реализуйте логику объединения узлов
        # Например: создаёте новый узел, переносите связи, удаляете старые
        print(f"Merging nodes: {list(self.selected_nodes)}")
        
        # После объединения очищаем выбор
        self.selected_nodes.clear()
        self.redraw()
        self.update_info_panel()

    def draw_static_legend(self):
        self.legend_canvas.delete("all")
        y_offset = 0
        for type_name, color in TYPE_COLORS.items():
            if type_name == "default": 
                continue
            label = TYPE_LABELS.get(type_name, type_name)
            self.legend_canvas.create_oval(10, y_offset + 5, 22, y_offset + 17, fill=color, outline=color, width=1)
            self.legend_canvas.create_text(35, y_offset + 11, text=label, fill=color, font=("Consolas", 10), anchor="w")
            y_offset += 25

    # -------------------------------------------------
    # GRAPH LOGIC
    # -------------------------------------------------

    def _create_layout(self):
        pos = nx.spring_layout(self.graph, seed=42, k=1.5, iterations=50)
        for node_id, (x, y) in pos.items():
            self.node_positions[node_id] = [800 + x * 500, 450 + y * 400]
        self.initial_positions = {k: v[:] for k, v in self.node_positions.items()}

    def toggle_sort(self):
        if self.is_sorted:
            self.node_positions = {k: v[:] for k, v in self.initial_positions.items()}
            self.is_sorted = False
            self.sort_btn.config(text="[ ↓ SORT_BY_TYPE ]")
        else:
            self._generate_sorted_layout()
            self.is_sorted = True
            self.sort_btn.config(text="[ ↑ UNSORT ]")
        self.redraw()

    def _generate_sorted_layout(self):
        type_groups = {}
        for node_id, attrs in self.graph.nodes(data=True):
            ntype = attrs["data"]["type"]
            type_groups.setdefault(ntype, []).append(node_id)

        new_pos = {}
        for ntype, nodes in type_groups.items():
            zone = TYPE_ZONES.get(ntype, {"x": 0, "y": 0})
            zone_cx = 800 + zone["x"]
            zone_cy = 450 + zone["y"]

            count = len(nodes)
            if count == 0: continue

            cols = math.ceil(math.sqrt(count))
            rows = math.ceil(count / cols)
            square_w = cols * CELL_SIZE
            square_h = rows * CELL_SIZE

            start_x = zone_cx - square_w / 2
            start_y = zone_cy - square_h / 2

            shuffled_nodes = nodes.copy()
            random.shuffle(shuffled_nodes)

            for idx, nid in enumerate(shuffled_nodes):
                row = idx // cols
                col = idx % cols
                
                cx = start_x + col * CELL_SIZE + CELL_SIZE / 2
                cy = start_y + row * CELL_SIZE + CELL_SIZE / 2
                
                jitter = CELL_SIZE / 5.0
                rx = random.uniform(-jitter, jitter)
                ry = random.uniform(-jitter, jitter)
                
                new_pos[nid] = [cx + rx, cy + ry]

        self.node_positions = new_pos

    # -------------------------------------------------
    # RENDERING
    # -------------------------------------------------

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
        self.add_bottom_hint()
        self.sort_btn.lift()

    def to_screen(self, wx, wy):
        return (wx + self.pan_offset[0]) * self.zoom, (wy + self.pan_offset[1]) * self.zoom

    def to_world(self, sx, sy):
        return sx / self.zoom - self.pan_offset[0], sy / self.zoom - self.pan_offset[1]

    def draw_nodes(self):
        r = NODE_RADIUS * self.zoom
        for node_id, attrs in self.graph.nodes(data=True):
            data = attrs["data"]
            x, y = self.screen_positions[node_id]
            selected = node_id in self.selected_nodes
            type_color = TYPE_COLORS.get(data["type"], TYPE_COLORS["default"])
            outline = SECONDARY_COLOR if selected else type_color
            width = 3 if selected else 2
            fill = BG_NODE 
            if selected:
                self.canvas.create_oval(x - r*1.2, y - r*1.2, x + r*1.2, y + r*1.2,
                                        outline=SECONDARY_COLOR, width=1, stipple="gray12")
            oval = self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                           fill=fill, outline=outline, width=width)
            if self.zoom >= 0.5:
                text_color = type_color if not selected else SECONDARY_COLOR
                text = self.canvas.create_text(x, y, text=data["name"], fill=text_color,
                                               width=60 * self.zoom,
                                               font=("Consolas", max(8, int(9 * self.zoom))))
                self.node_items[text] = node_id
            self.node_items[oval] = node_id

    def _get_edge_end_point(self, x1, y1, x2, y2, radius):
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0: return x2, y2
        nx, ny = dx / length, dy / length
        return x2 - nx * radius, y2 - ny * radius

    def draw_edges(self):
        edge_groups = {}
        for u, v, key, attrs in self.graph.edges(keys=True, data=True):
            edge_data = attrs["data"]
            pair = tuple(sorted([u, v]))
            edge_groups.setdefault(pair, {"forward": [], "backward": []})
            if edge_data.get("source") == u:
                edge_groups[pair]["forward"].append((u, v, edge_data))
            else:
                edge_groups[pair]["backward"].append((v, u, edge_data))

        for pair, groups in edge_groups.items():
            for idx, (u, v, edge_data) in enumerate(groups["forward"]):
                total = len(groups["forward"]) + len(groups["backward"])
                self._draw_curved_edge(u, v, edge_data, 1, idx, total)
            for idx, (u, v, edge_data) in enumerate(groups["backward"]):
                total = len(groups["forward"]) + len(groups["backward"])
                self._draw_curved_edge(u, v, edge_data, -1, len(groups["forward"]) + idx, total)

    def _draw_curved_edge(self, u, v, edge_data, direction=1, index=0, total_edges=1):
        sx1, sy1 = self.screen_positions[u]
        sx2, sy2 = self.screen_positions[v]

        dx = sx2 - sx1
        dy = sy2 - sy1
        length = math.hypot(dx, dy) or 1
        nx, ny = dx / length, dy / length
        px, py = -ny, nx

        mx, my = (sx1 + sx2) / 2, (sy1 + sy2) / 2

        base_curve = 60 * direction * self.zoom
        spread = (index - (total_edges - 1) / 2) * 40 * self.zoom
        
        cx = mx + px * (base_curve + spread)
        cy = my + py * (base_curve + spread)

        r = NODE_RADIUS * self.zoom
        start_x, start_y = self._get_edge_end_point(sx2, sy2, sx1, sy1, r)
        end_x, end_y = self._get_edge_end_point(sx1, sy1, sx2, sy2, r)

        edge_id = edge_data.get("id")
        is_selected = edge_id in self.selected_edges
        
        color = SELECT_COLOR if is_selected else SECONDARY_COLOR
        width = EDGE_WIDTH_SELECTED * self.zoom if is_selected else EDGE_WIDTH_NORMAL * self.zoom

        arrow_len = max(8, 12 * self.zoom)
        arrow_wid = max(8, 14 * self.zoom)
        arrow_off = max(2, 4 * self.zoom)

        line_id = self.canvas.create_line(start_x, start_y, cx, cy, end_x, end_y,
                                          smooth=True, splinesteps=12,
                                          fill=color, width=width,
                                          arrow=tk.LAST,
                                          arrowshape=(arrow_len, arrow_wid, arrow_off))

        relation = edge_data.get("relation", "")
        if relation and self.zoom >= 0.6:
            label_x = cx + px * 15 * self.zoom * direction
            label_y = cy + py * 15 * self.zoom * direction

            label_id = self.canvas.create_text(label_x, label_y, text=relation,
                                               fill=SECONDARY_COLOR,
                                               font=("Consolas", max(8, int(9 * self.zoom)), "bold"))
            
            bbox = self.canvas.bbox(label_id)
            if bbox:
                pad_x = 6 * self.zoom
                pad_y = 3 * self.zoom
                bg_id = self.canvas.create_rectangle(
                    bbox[0]-pad_x, bbox[1]-pad_y,
                    bbox[2]+pad_x, bbox[3]+pad_y,
                    fill=BG_DARK, outline=SECONDARY_COLOR, width=1
                )
                self.canvas.tag_lower(bg_id, label_id)
                self.edge_labels[edge_id] = {"text": label_id, "bg": bg_id}

        self.edge_items[line_id] = {
            "id": edge_id, "source": u, "target": v,
            "relation": relation, "description": edge_data.get("description", ""),
            "canvas_id": line_id
        }

    # -------------------------------------------------
    # INTERACTION
    # -------------------------------------------------

    def on_press(self, event):
        if self.panning: return
        shift = (event.state & 0x0001) != 0

        halo = max(10, 15 * self.zoom)
        closest = self.canvas.find_closest(event.x, event.y, halo=halo)
        
        clicked_node = None
        clicked_edge = None
        for item in closest:
            if item in self.node_items:
                clicked_node = self.node_items[item]
                break
            elif item in self.edge_items:
                clicked_edge = self.edge_items[item]
                break

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
            
            self.update_info_panel()
            self.redraw()
            return

        if clicked_edge:
            edge_id = clicked_edge["id"]
            if shift: 
                self.selected_edges.add(edge_id)
            else:
                self.selected_edges = {edge_id}
                self.selected_nodes.clear()
                self.is_editing = False
            
            self.update_info_panel()
            self.redraw()
            return

        if not shift:
            self.selected_nodes.clear()
            self.selected_edges.clear()
            self.is_editing = False
            self.update_info_panel()
            self.redraw()

    def on_drag(self, event):
        if not self.dragging_node or self.panning: return

        new_sx = event.x - self.drag_offset_x
        new_sy = event.y - self.drag_offset_y
        wx, wy = self.to_world(new_sx, new_sy)
        
        self.node_positions[self.dragging_node] = [wx, wy]
        self.redraw()

    def on_release(self, event):
        self.dragging_node = None

    def on_pan_start(self, event):
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="fleur")

    def on_pan(self, event):
        if not self.panning: return
        
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

    def on_zoom(self, event):
        self._zoom_event(event.delta)

    def _zoom_event(self, delta):
        if delta == 0: return
        factor = 1.15 if delta > 0 else 0.85

        wx, wy = self.to_world(self.last_mouse_x, self.last_mouse_y)
        self.zoom *= factor
        self.zoom = max(0.15, min(5.0, self.zoom))

        self.pan_offset[0] = self.last_mouse_x / self.zoom - wx
        self.pan_offset[1] = self.last_mouse_y / self.zoom - wy
        self.redraw()

    def _track_mouse(self, event):
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def run(self):
        self.root.mainloop()