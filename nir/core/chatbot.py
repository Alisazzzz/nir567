#All stuff for console chatbot is here



#--------------------------
#---------imports----------
#--------------------------

import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich import box
from rich.align import Align
from rich.live import Live
from rich.text import Text

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from nir.data import loader
from nir.core.answers_generator import generate_answer_based_on_plan, generate_plan
from nir.core.context_retriever import form_context_with_llm, form_context_without_llm
from nir.embedding.vector_store_loader import VectorStoreInfo
from nir.graph.graph_construction import create_embeddings, extract_graph, get_next_chunk_id, update_embeddings, update_graph
from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.llm.manager import ModelManager
from nir.llm.providers import ModelConfig
from nir.core.chat_history import ChatHistory



#---------------------------------
#---------additional stuff--------
#---------------------------------

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chunks = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    token_count = 0
    for chunk in chunks:
        length = len(chunk)
        if re.match(r"[A-Za-z0-9]+$", chunk):
            token_count += max(1, round(length / 3.5))
        elif re.match(r"[А-Яа-яЁё]+$", chunk):
            token_count += max(1, round(length / 2.4))
        else:
            token_count += 1
    return token_count



#-------------------------------
#---------theme & colors--------
#-------------------------------

console = Console()

class ColorTheme:
    PRIMARY = "#0fc23c"
    SECONDARY = "#04681d"
    ACCENT = "#ffaa00"
    SUCCESS = "#1cff08"
    DANGER = "#ff4444"
    WARNING = "#eeff00"
    INFO = "#00aeff"
    TEXT = "#D1D1D1"
    DARK = "#0a0a0a"
    


#-------------------------------
#---------data classes----------
#-------------------------------

@dataclass
class ModelInfo:
    name: str
    model_name: str
    option: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    has_api_key: bool = False
    
    def display_string(self) -> str:
        return f"{self.name} ({self.model_name}, option: {self.option}, temperature: {self.temperature})"
    
    @staticmethod
    def from_dict(data: dict) -> 'ModelInfo':
        return ModelInfo(
            name=data['name'],
            model_name=data['model_name'],
            option=data['option'],
            temperature=data.get('temperature'),
            max_tokens=data.get('max_tokens'),
            has_api_key=data.get('has_api_key', False)
        )



#-----------------------------
#-------menu components-------
#-----------------------------

class MenuItem:
    def __init__(self, title: str, action=None, data=None):
        self.title = title
        self.action = action
        self.data = data

class InteractiveMenu:
    def __init__(self, title: str, items: List[MenuItem], show_back: bool = True, description: str = ""):
        self.title = title
        self.items = items
        self.selected_index = 0
        self.show_back = show_back
        self.description = description
    
    def display(self) -> Optional[MenuItem]:
        while True:
            ConsoleUI.clear()
            ConsoleUI.print_header(self.title)
            if self.description:
                console.print(f"\n[dim italic]{self.description}[/dim italic]\n")
            console.print(f"\n[bold {ColorTheme.INFO}]Use ↑/↓ to navigate, Enter to select, Esc to cancel[/bold {ColorTheme.INFO}]\n")
            for idx, item in enumerate(self.items):
                if idx == self.selected_index:
                    prefix = "→ "
                    color = ColorTheme.PRIMARY
                else:
                    prefix = "  "
                    color = ColorTheme.TEXT
                console.print(f"{prefix}[{color}]{item.title}[/{color}]")
            if self.show_back:
                back_idx = len(self.items)
                if back_idx == self.selected_index:
                    console.print(f"→ [red]← Back[/red]")
                else:
                    console.print(f"  [red]← Back[/red]")
            
            key = ConsoleUI.get_key()
            if key == 'up':
                self.selected_index = (self.selected_index - 1) % (len(self.items) + (1 if self.show_back else 0))
            elif key == 'down':
                self.selected_index = (self.selected_index + 1) % (len(self.items) + (1 if self.show_back else 0))
            elif key == 'enter':
                if self.show_back and self.selected_index == len(self.items):
                    return None
                elif self.selected_index < len(self.items):
                    return self.items[self.selected_index]
            elif key == 'escape':
                return None

class ConsoleUI:
    @staticmethod
    def clear():
        os.system('clear' if os.name == 'posix' else 'cls')
    
    @staticmethod
    def print_header(text: str, color: str = ColorTheme.PRIMARY):
        console.print(Panel(Align.center(text, style=color), border_style=ColorTheme.SECONDARY, box=box.SQUARE))

    @staticmethod
    def get_key():
        if sys.platform == 'win32':
            import msvcrt
            try:
                key = msvcrt.getch()
                if key == b'\xe0':
                    key = msvcrt.getch()
                    if key == b'H':
                        return 'up'
                    elif key == b'P':
                        return 'down'
                    elif key == b'M':
                        return 'right'
                    elif key == b'K':
                        return 'left'
                elif key == b'\r':
                    return 'enter'
                elif key == b'\x1b':
                    return 'escape'
                elif key == b'\x00':
                    key2 = msvcrt.getch()
                    if key2 == b'S':
                        return 'alt+s'
                return None
            except:
                return None
        else:
            try:
                import termios
                import tty
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':
                        ch = sys.stdin.read(2)
                        if ch == '[A':
                            return 'up'
                        elif ch == '[B':
                            return 'down'
                        elif ch == '[C':
                            return 'right'
                        elif ch == '[D':
                            return 'left'
                    elif ch == '\r':
                        return 'enter'
                    elif ch == '\x1b':
                        return 'escape'
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except:
                return None
        return None
    
    @staticmethod
    def get_simple_input(prompt, valid_options=None):
        while True:
            value = input(prompt)
            if valid_options and value not in valid_options:
                ConsoleUI.print_error(f"Invalid option. Choose from: {', '.join(valid_options)}")
                continue
            return value
         
    @staticmethod
    def print_success(text: str):
        console.print(f"[{ColorTheme.SUCCESS}][/{ColorTheme.SUCCESS}] {text}")
    
    @staticmethod
    def print_error(text: str):
        console.print(f"[{ColorTheme.DANGER}][/{ColorTheme.DANGER}] {text}")
    
    @staticmethod
    def print_info(text: str):
        console.print(f"[{ColorTheme.INFO}][/{ColorTheme.INFO}] {text}")
    
    @staticmethod
    def print_warning(text: str):
        console.print(f"[{ColorTheme.WARNING}][/{ColorTheme.WARNING}] {text}")
    
    @staticmethod
    def print_step(step: int, total: int, text: str):
        console.print(f"\n[{ColorTheme.ACCENT}][{step}/{total}][/{ColorTheme.ACCENT}] {text}")



#-----------------------------
#-------form components-------
#-----------------------------

class FormField:
    def __init__(self, label: str, default: Any = "", required: bool = True, field_type: str = "text", options: List[str] = None, help_text: str = ""):
        self.label = label
        self.default = default
        self.required = required
        self.field_type = field_type
        self.options = options
        self.help_text = help_text
        self.value = default
        self.error = ""

class Form:
    def __init__(self, title: str, fields: List[FormField], clear_screen: bool = True):
        self.title = title
        self.fields = fields
        self.selected_field = 0
        self.editing = False
        self.clear_screen = clear_screen
        self.first_render = True
    
    def display(self) -> Optional[Dict[str, Any]]:
        while True:
            if self.clear_screen:
                ConsoleUI.clear()
                ConsoleUI.print_header(self.title)
                self.render_fields()
            else:
                if self.first_render:
                    ConsoleUI.print_header(self.title)
                    self.render_fields()
                    self.first_render = False
                else:
                    if self.editing:
                        console.print(f"\n[bold {ColorTheme.SECONDARY}]Editing: {self.fields[self.selected_field].label}[/bold {ColorTheme.SECONDARY}]")
                        console.print("[dim]Type your value and press Enter. Esc to cancel editing.[/dim]")
                    else:
                        console.print("\n" + " " * 50 + "\r", end="")
                        console.print("[dim]Enter: edit field | Esc: cancel | Alt+S: submit[/dim]", end="")
            
            key = ConsoleUI.get_key()
            if key == 'up' and not self.editing:
                self.selected_field = (self.selected_field - 1) % len(self.fields)
                if not self.clear_screen:
                    self.update_selection()
            elif key == 'down' and not self.editing:
                self.selected_field = (self.selected_field + 1) % len(self.fields)
                if not self.clear_screen:
                    self.update_selection()
            elif key == 'enter' and not self.editing:
                self.editing = True
                self.fields[self.selected_field].error = ""
                if not self.clear_screen:
                    console.print(f"\n[bold green]Editing: {self.fields[self.selected_field].label}[/bold green]")
                    console.print("[dim]Type your value and press Enter. Esc to cancel editing.[/dim]")
                self.submit_current_field()
            elif key == 'escape':
                if self.editing:
                    self.editing = False
                    self.fields[self.selected_field].value = self.fields[self.selected_field].default
                    self.fields[self.selected_field].error = ""
                    if not self.clear_screen:
                        self.update_field_value()
                else:
                    return None
            elif key == 'alt+s' and not self.editing:
                if self.validate_form():
                    return self.get_results()
    
    def render_fields(self):
        console.print(f"\n[{ColorTheme.INFO}]Press Enter to edit field, ↑/↓ to navigate[/{ColorTheme.INFO}]\n")
        for idx, field in enumerate(self.fields):
            prefix = "→ " if idx == self.selected_field and not self.editing else "  "
            
            if field.field_type == "boolean":
                value_display = "✅ Yes" if field.value else "❎ No"
            elif field.field_type == "select" and field.options:
                value_display = f"[{field.value}]" if field.value else "[Not selected]"
            else:
                value_display = field.value if field.value else "[empty]"

            required_mark = "[red]*[/red]" if field.required else "  "
            if field.error:
                console.print(f"{prefix}[yellow]{field.label}:[/yellow] [red]{field.error}[/red]")
            else:
                console.print(f"{prefix}{required_mark} [cyan]{field.label}:[/cyan] [green]{value_display}[/green]")
        
        console.print(f"\n[dim]Enter: edit field | Esc: cancel | Alt+S: submit[/dim]")
    
    def update_selection(self):
        console.print("\n" * (len(self.fields) + 3), end="")
        self.render_fields()
    
    def update_field_value(self):
        console.print("\n" * (len(self.fields) + 3), end="")
        self.render_fields()

    def submit_current_field(self):
        field = self.fields[self.selected_field]
        console.print(f"\n[bold]Enter value for {field.label}:[/bold]")
        if field.default:
            console.print(f"[dim]Default: {field.default}[/dim]")
        user_input = Prompt.ask("", default=str(field.value) if field.value else "")
        
        try:
            if field.field_type == "number":
                value = int(user_input) if user_input else (field.default if field.default else 0)
            elif field.field_type == "float":
                value = float(user_input) if user_input else (field.default if field.default else 0.0)
            elif field.field_type == "boolean":
                value = user_input.lower() in ['yes', 'y', 'true', '1', 'да'] if user_input else field.default
            elif field.field_type == "select" and field.options:
                if user_input in field.options:
                    value = user_input
                elif user_input:
                    field.error = f"Invalid option. Choose from: {', '.join(field.options)}"
                    return
                else:
                    value = field.default
            else:
                value = user_input if user_input else field.default
            
            if field.required and not value:
                field.error = "This field is required"
                return
            
            field.value = value
            field.error = ""
            self.editing = False
            
        except ValueError:
            field.error = f"Invalid {field.field_type} value"
    
    def validate_form(self) -> bool:
        valid = True
        for field in self.fields:
            if field.required and not field.value:
                field.error = "This field is required"
                valid = False
        return valid
    
    def get_results(self) -> Dict[str, Any]:
        import re
        result = {}
        for field in self.fields:
            key = field.label.lower()
            key = re.sub(r'[^a-z0-9_]', '_', key)
            key = re.sub(r'_+', '_', key)
            key = key.strip('_')
            result[key] = field.value
        return result



#----------------------------
#------chat application------
#----------------------------

class ChatApplication:
    def __init__(self):
        self.model_manager = ModelManager()
        self.graph: Optional[KnowledgeGraph] = None
        self.chat_model: Optional[BaseLanguageModel] = None
        self.instruct_model: Optional[BaseLanguageModel] = None
        self.embedding_model: Optional[Embeddings] = None
        self.chat_model_info: Optional[ModelInfo] = None
        self.instruct_model_info: Optional[ModelInfo] = None
        self.embedding_model_info: Optional[ModelInfo] = None
        self.graph_filepath: Optional[str] = None
        self.add_history = False
        self.chat_history: Optional[ChatHistory] = None
        self.language: str = "en"
    
    def run(self):
        self.intro()
        if self.intro() is False:
            return
        if not self.select_instruct_model():
            return
        if not self.select_chat_model():
            return
        self.select_or_create_graph()
        self.main_chat_loop()

    
    def intro(self):
        ConsoleUI.clear()
        ConsoleUI.print_header("NIR NIR INSTRUMENT", ColorTheme.PRIMARY)
        welcome_text = """
Welcome to the NIR NIR System for narrative elements generation!

This application allows you to:
- Extract knowledge graphs from documents
- Query the graph using natural language
- Update the graph based on conversations
- Visualize and edit the graph interactively

Let's set up your environment!
        """
        console.print(Panel(welcome_text, border_style=ColorTheme.SECONDARY, box=box.SQUARE))
        Prompt.ask("\nPress Enter to continue", default="")
    
    def select_instruct_model(self) -> bool:
        ConsoleUI.print_step(1, 5, "Selecting Graph Extraction Model")
        console.print("\n[italic]This model should be strict and precise. Use instruct models with temperature 0.0.[/italic]\n")  
        result = self.select_chat_model_from_manager("Graph Extraction Model")
        if result is None:
            console.print("\n[yellow]Model selection cancelled. Exiting...[/yellow]")
            return False
        self.instruct_model_info = result
        self.instruct_model = self.model_manager.get_chat_model(self.instruct_model_info.name)
        ConsoleUI.print_success(f"Model '{self.instruct_model_info.name}' loaded successfully")
        return True
    
    def select_chat_model(self) -> bool:
        ConsoleUI.print_step(2, 5, "Selecting Chat Model")
        console.print("\n[italic]This model should be creative. Use temperature around 0.7.[/italic]\n")
        result = self.select_chat_model_from_manager("Chat Model")
        if result is None:
            console.print("\n[yellow]Model selection cancelled. Exiting...[/yellow]")
            return False
        self.chat_model_info = result
        self.chat_model = self.model_manager.get_chat_model(self.chat_model_info.name)
        ConsoleUI.print_success(f"Model '{self.chat_model_info.name}' loaded successfully")
        return True
    
    def select_chat_model_from_manager(self, model_type: str) -> Optional[ModelInfo]:
        while True:
            models = self.model_manager.list_chat_models()
            if not models:
                ConsoleUI.print_warning("No models found. Let's create one!")
                if not Confirm.ask("\nCreate a new model?", default=True):
                    return None
                self.create_new_chat_model()
                continue

            items = []
            for model in models:
                title = ModelInfo.from_dict(model).display_string()
                items.append(MenuItem(title, data=model))
            items.append(MenuItem("Create new model", action="create"))
            if (model_type=="Graph Extraction Model"):
                description = "Select model for GRAPH EXTRACTION (press Esc to cancel)"
            else:
                description = "Select model for CHAT (press Esc to cancel)"

            menu = InteractiveMenu(f"Select {model_type}", items, show_back=False, description=description)
            selected = menu.display()
            if selected is None:
                return None
            if selected.action == "create" or selected.data is None:
                self.create_new_chat_model()
            else:
                model = selected.data
                return ModelInfo(
                    name=model['name'],
                    model_name=model['model_name'],
                    option=model['option'],
                    temperature=model.get('temperature'),
                    max_tokens=model.get('max_tokens'),
                    has_api_key=model.get('has_api_key', False)
                )

    def select_embedding_model(self) -> Optional[ModelInfo]:
        while True:
            models = self.model_manager.list_embedding_models()
            if not models:
                ConsoleUI.print_warning("No embedding models found. Let's create one!")
                if not Confirm.ask("\nCreate a new model?", default=True):
                    return None
                self.create_new_embedding_model()
                continue
    
            items = []
            for model in models:
                title = f"{model['name']} ({model['model_name']}, {model['option']})"
                items.append(MenuItem(title, data=model))
            items.append(MenuItem("Create new model", action="create"))
            
            menu = InteractiveMenu("Select Embedding Model", items, show_back=True, description="Press Esc to cancel")
            selected = menu.display()
            if selected is None:
                return None
            if selected.action == "create" or selected.data is None:
                self.create_new_embedding_model()
            else:
                model = selected.data
                return ModelInfo(
                    name=model['name'],
                    model_name=model['model_name'],
                    option=model['option'],
                    has_api_key=model.get('has_api_key', False)
                )
    
    def create_new_chat_model(self):
        fields = [
            FormField("Model title (for your understanding): ", required=True),
            FormField("Model name (official name)", required=True),
            FormField("Option", default="ollama", required=True, options=["ollama", "openai", "hf_local", "hf_api"]),
            FormField("Temperature", default="0.7", field_type="float"),
            FormField("Max Tokens", default="2048", field_type="number"),
            FormField("API Key (if needed)", required=False),
        ]
        form = Form("Create New Chat Model", fields)
        result = form.display()
        if result is None:
            return
        config = ModelConfig(
            model_name=result["model_name_official_name"],
            temperature=result["temperature"],
            max_tokens=result["max_tokens"]
        )
        api_key = result.get("api_key_if_needed")
        if api_key == "":
            api_key = None
        self.model_manager.create_chat_model(
            name=result["model_title_for_your_understanding"],
            option=result["option"],
            config=config,
            api_info=api_key
        )
        ConsoleUI.print_success(f"Model '{result['model_title_for_your_understanding']}' created successfully")
        Prompt.ask("\nPress Enter to continue", default="")
    
    def create_new_embedding_model(self):
        fields = [
            FormField("Model title (for your understanding)", required=True),
            FormField("Model name (official name)", required=True),
            FormField("Option", default="ollama", required=True, options=["ollama", "openai", "hf_local"]),
            FormField("API Key (if needed)", required=False),
        ]
        form = Form("Create New Embedding Model", fields)
        result = form.display()
        if result is None:
            return
        api_key = result.get("api_key_if_needed")
        if api_key == "":
            api_key = None
        self.model_manager.create_embedding_model(
            name=result["model_title_for_your_understanding"],
            option=result["option"],
            model_name=result["model_id_official_name"],
            api_info=api_key
        )
        ConsoleUI.print_success(f"Embedding model '{result['model_title_for_your_understanding']}' created successfully")
        Prompt.ask("\nPress Enter to continue", default="")
    
    def select_existing_graph(self) -> bool:
        graphs_dir = Path("assets/graphs")
        if not graphs_dir.exists():
            ConsoleUI.print_error("No graphs directory found")
            Prompt.ask("Press Enter to continue", default="")
            return False
        graph_files = list(graphs_dir.glob("*.json"))
        if not graph_files:
            ConsoleUI.print_warning("No graph files found in assets/graphs")
            Prompt.ask("Press Enter to continue", default="")
            return False
        
        items = [MenuItem(f.name, data=str(graphs_dir / f.name)) for f in graph_files]
        menu = InteractiveMenu("Select Graph", items, show_back=True, description="Press Esc to cancel")
        selected = menu.display()
        
        if selected and selected.data:
            self.graph_filepath = selected.data
            try:
                self.graph = NetworkXGraph()
                self.graph.load(path=self.graph_filepath) 
                embedding_model_name = self.graph.get_embedding_model()  
                self.embedding_model = self.model_manager.get_embedding_model(embedding_model_name)
                if not self.embedding_model:
                    console.print("\n[yellow]Embedding model needed for this graph[/yellow]")
                    embedding_info = self.select_embedding_model()
                    if embedding_info is None:
                        ConsoleUI.print_error("Embedding model selection cancelled. Cannot load graph.")
                        Prompt.ask("Press Enter to continue", default="")
                        return False
                    self.embedding_model = self.model_manager.get_embedding_model(embedding_info.name)
                    self.embedding_model_info = embedding_info
                
                ConsoleUI.print_success(f"Graph '{selected.title}' loaded successfully")
                Prompt.ask("\nPress Enter to continue", default="")
                return True
            except Exception as e:
                ConsoleUI.print_error(f"Failed to load graph: {e}")
                Prompt.ask("Press Enter to continue", default="")
                return False
        
        return False

    def select_or_create_graph(self):
        ConsoleUI.print_step(3, 5, "Graph Setup")
        items = [
            MenuItem("Select existing graph", action="select"),
            MenuItem("Create new graph", action="create")
        ]
        menu = InteractiveMenu("Graph Setup", items, show_back=False, description="Press Esc to cancel setup")
        selected = menu.display()
        if selected:
            if selected.action == "select":
                self.select_existing_graph()
            elif selected.action == "create":
                self.create_new_graph()

    def create_new_graph(self) -> bool:
        ConsoleUI.clear()
        ConsoleUI.print_header("Create New Graph")
        embedding_info = self.select_embedding_model()
        if embedding_info is None:
            ConsoleUI.print_error("Embedding model selection cancelled. Graph creation aborted.")
            Prompt.ask("\nPress Enter to continue", default="")
            return False
        self.embedding_model = self.model_manager.get_embedding_model(embedding_info.name)
        self.embedding_model_info = embedding_info

        fields = [
            FormField("Text file (from assets/documents)", required=True),
            FormField("Chunk size", default="500", field_type="number"),
            FormField("Chunk overlap", default="50", field_type="number"),
            FormField("Language", default="en", required=True),
            FormField("Output graph name", required=True),
        ]
        
        form = Form("Graph Configuration", fields)
        result = form.display()
        
        if result is None:
            return False
        
        text_path = Path(f"assets/documents/{result['text_file_from_assets_documents']}")
        if not text_path.exists():
            ConsoleUI.print_error(f"File {text_path} not found")
            Prompt.ask("\nPress Enter to continue", default="")
            return False
        
        columns_to_use = None
        if text_path.suffix == '.csv':
            ConsoleUI.clear()
            ConsoleUI.print_header("CSV Columns Configuration")
            try:
                import pandas as pd
                df_sample = pd.read_csv(text_path, nrows=1)
                console.print("\n[bold]Available columns in CSV:[/bold]")
                for idx, col in enumerate(df_sample.columns, 1):
                    console.print(f"  {idx}. [green]{col}[/green]")
                console.print("")
            except Exception as e:
                console.print(f"[{ColorTheme.WARNING}]⚠ Could not preview CSV: {e}[/{ColorTheme.WARNING}]\n")
            csv_fields = [FormField("Columns to use (comma-separated)", required=False)]
            csv_form = Form("Enter columns (leave empty for all)", csv_fields, clear_screen=False)
            csv_result = csv_form.display()
            
            if csv_result is None:
                return False
            
            columns_input = csv_result.get("columns_to_use_comma_separated", "").strip()
            if columns_input:
                columns_to_use = [col.strip() for col in columns_input.split(",")]
                ConsoleUI.print_success(f"Using columns: {', '.join(columns_to_use)}")
            else:
                ConsoleUI.print_info("Using all columns from CSV")
                columns_to_use = None 
            Prompt.ask("\nPress Enter to continue", default="")

        if text_path.suffix == '.csv':
            if columns_to_use:
                data = loader.loadCSV_withColumns(path=str(text_path), columns=columns_to_use)
            else:
                data = loader.loadCSV(path=str(text_path))
        elif text_path.suffix == '.pdf':
            data = loader.loadPDF(str(text_path))
        else:
            data = loader.loadTXT(str(text_path))
        
        chunks = loader.to_chunk_unique_id(docs=data, start_chunk_id=0, chunk_size=int(result["chunk_size"]), chunk_overlap=int(result["chunk_overlap"]))   
        self.graph = extract_graph(chunks=chunks, llm=self.instruct_model, embedding_model=self.embedding_model, graph_class=NetworkXGraph, preserve_all_data=False, language=result["language"]) 
        self.language = result["language"]
        vector_db_info = VectorStoreInfo(
            type="chromadb",
            info={
                "name": f"{result['output_graph_name']}_db",
                "path": "assets/databases/chroma_db"
            }
        )
        self.graph.create_vector_db(vector_db_info)
        create_embeddings(self.graph, self.graph.get_vector_db(), self.embedding_model)
        self.graph.set_embedding_model(embedding_info.name) 

        output_path = f"assets/graphs/{result['output_graph_name']}.json"
        self.graph.save(path=output_path)
        self.graph_filepath = output_path
        
        ConsoleUI.print_success(f"Graph created and saved to {output_path}")
        Prompt.ask("\nPress Enter to continue", default="")
        return True
    
    def main_chat_loop(self):
        ConsoleUI.print_step(4, 5, "Ready to Chat!")
        while True:
            ConsoleUI.clear()
            ConsoleUI.print_header("NIR NIR Chat System!")

            status_text = f"""
    [bold {ColorTheme.PRIMARY}]Current Configuration:[/bold {ColorTheme.PRIMARY}]
    - Graph: [cyan]{Path(self.graph_filepath).name if self.graph_filepath else 'None'}[/cyan]
    - Chat Model: [cyan]{self.chat_model_info.display_string()}[/cyan]
    - Extraction Model: [cyan]{self.instruct_model_info.display_string()}[/cyan]
            """
            main_menu_items = [
                MenuItem("Start Chat Session", action="start_chat"),
                MenuItem("Change Graph", action="change_graph"),
                MenuItem("Change Chat Model", action="change_chat_model"),
                MenuItem("Change Extraction Model", action="change_extraction_model"),
                MenuItem("Visualize & Edit Graph", action="visualize"),
                MenuItem("Exit", action="exit"),
            ]
            menu = InteractiveMenu(
                title="Main Menu",
                items=main_menu_items,
                show_back=False
            )

            selected = menu.display()
            if selected is None:
                continue

            if selected.action == "start_chat":
                self.run_chat_session()
            elif selected.action == "change_graph":
                if self.select_or_create_graph():
                    ConsoleUI.print_success("Graph changed successfully!")
                    Prompt.ask("\nPress Enter to continue", default="")
            elif selected.action == "change_chat_model":
                if self.select_chat_model():
                    ConsoleUI.print_success("Chat model changed successfully!")
                    Prompt.ask("\nPress Enter to continue", default="")
                else:
                    ConsoleUI.print_warning("Chat model selection cancelled.")
                    Prompt.ask("\nPress Enter to continue", default="")
            elif selected.action == "change_extraction_model":
                if self.select_instruct_model():
                    ConsoleUI.print_success("Extraction model changed successfully!")
                    Prompt.ask("\nPress Enter to continue", default="")
                else:
                    ConsoleUI.print_warning("Extraction model selection cancelled.")
                    Prompt.ask("\nPress Enter to continue", default="")
            elif selected.action == "toggle_history":
                self.add_history = not self.add_history
                status = "enabled" if self.add_history else "disabled"
                ConsoleUI.print_success(f"Conversation history {status}")
                Prompt.ask("\nPress Enter to continue", default="")
            elif selected.action == "visualize":
                self.visualize_graph()
            elif selected.action == "exit":
                if self.exit_app():
                    break

    def run_chat_session(self):
        ConsoleUI.clear()
        ConsoleUI.print_header("Interactive Chat Session")
        
        info_text = f"""
    [dim]Graph: {Path(self.graph_filepath).name}[/dim]
    [dim]Chat model: {self.chat_model_info.display_string()}[/dim]
    [dim]Add world history to context: {'ON (add event sequence to context)' if self.add_history else 'OFF (do not add event sequence to context)'}[/dim]

    [bold green]Commands:[/bold green]
    - [cyan]/quit[/cyan] or [cyan]/q[/cyan] - Exit to main menu
    - [cyan]/update[/cyan] - Update graph with last answer
    - [cyan]/history[/cyan] - Toggle add world history to context (think if event sequence of world history is needed)
    - [cyan]/clear[/cyan] - Clear all current chat history, reset topic and start working on new element without any impact of previous messages
    
    [bold]Just type your question to chat with the graph![/bold]
    """
        console.print(Panel(info_text, border_style=ColorTheme.SECONDARY, box=box.SQUARE))
        session_add_history = self.add_history

        if self.graph_filepath:
            history_dir = Path("assets/chats")
            history_dir.mkdir(parents=True, exist_ok=True)
            graph_name = Path(self.graph_filepath).stem
            history_path = history_dir / f"{graph_name}_chat_history.json"
            if history_path.exists():
                self.chat_history = ChatHistory.load(str(history_path))
            else:
                self.chat_history = ChatHistory(
                    graph_path=self.graph_filepath,
                    file_path=str(history_path)
                )
        else:
            self.chat_history = None

        current_query = ""
        current_message_history = ""
        last_answer = ""

        while True:
            console.print("\n[bold cyan]" + "─" * 50 + "[/bold cyan]")
            query = Prompt.ask(f"\n[{ColorTheme.INFO}]Your request:[/{ColorTheme.INFO}]")

            if query.lower() in ['/quit', '/q']:
                console.print(f"\n[{ColorTheme.SECONDARY}]Returning to main menu...[/{ColorTheme.SECONDARY}]")
                break
            elif query.lower() == '/clear':
                current_query = ""
                current_message_history = ""
                console.print(f"[{ColorTheme.SECONDARY}]Local session history cleared![/{ColorTheme.SECONDARY}]")
                continue
            elif query.lower() == '/update':
                if last_answer != "":
                    self.update_graph_with_response(last_answer)
                else:
                    console.print(f"[{ColorTheme.SECONDARY}]⚠ No answer to update yet. Ask a question first![/{ColorTheme.SECONDARY}]")
                continue
            elif query.lower() == '/history':
                session_add_history = not session_add_history
                status = "ON" if session_add_history else "OFF"
                console.print(f"[{ColorTheme.SECONDARY}]Conversation history turned {status}[/{ColorTheme.SECONDARY}]")
                continue

            status_text = Text("Initializing...", style=f"{ColorTheme.INFO}")
            
            with Live(status_text, refresh_per_second=10, transient=True) as live:
                try:
                    current_query += f"\n{query}"

                    status_text = Text("Searching graph...", style=f"{ColorTheme.INFO}")
                    live.update(status_text)
                    context_from_graph = form_context_with_llm(
                        query=current_query,
                        graph=self.graph,
                        llm=self.instruct_model,
                        embedding_model=self.embedding_model,
                        add_history=session_add_history,
                        max_tokens=2048
                    )

                    final_context = ""
                    if current_message_history:
                        final_context += f"CHAT HISTORY: {current_message_history}\n\n"
                    final_context += f"RETRIEVED KNOWLEDGE FROM GRAPH:\n{context_from_graph}"
                    
                    status_text = Text("Generating plan...", style=f"{ColorTheme.INFO}")
                    live.update(status_text)
                    plan = generate_plan(query, final_context, self.chat_model, language=self.language)
                    
                    status_text = Text("Writing answer...", style=f"{ColorTheme.INFO}")
                    live.update(status_text)
                    answer = generate_answer_based_on_plan(query, plan, final_context, self.chat_model, language=self.language)

                    last_answer = answer

                    if self.chat_history:
                        self.chat_history.add_message_to_history("user", query)
                        self.chat_history.add_message_to_history("assistant", answer)
                        current_message_history += f"\nUser: \n{query}" 
                        current_message_history += f"\nModel: \n{answer}"
                        if estimate_tokens(current_message_history) > 2048:
                            current_message_history = self.chat_history.get_context_window(max_tokens=2048) 
                        self.chat_history.save()

                except Exception as e:
                    console.print(f"\n[{ColorTheme.DANGER}]Error: {e}[/{ColorTheme.DANGER}]")
                    if not Confirm.ask("\nContinue chatting?", default=True):
                        break
                    continue

            ConsoleUI.print_header("Answer")
            console.print(Panel(Markdown(answer), border_style=ColorTheme.PRIMARY, box=box.SQUARE))
            if last_answer:
                console.print("[dim]Tip: Type /update to add this answer to the graph[/dim]")
    
    def update_graph_with_response(self, response: str):
        greatest_id = get_next_chunk_id(self.graph)
        data = loader.convertFromString(response)
        chunks = loader.to_chunk_unique_id(docs=data, start_chunk_id=greatest_id)
        update_graph(chunks, self.instruct_model, self.embedding_model, self.graph)
        
        if self.graph.get_vector_db():
            update_embeddings(self.graph, self.graph.get_vector_db(), self.embedding_model)
        
        self.graph.save(path=self.graph_filepath)
        
        ConsoleUI.print_success("Graph updated successfully")
    
    def visualize_graph(self):
        ConsoleUI.print_info("Launching graph visualization...")
        try:
            from nir.graph.gui.graph_window import GraphWindow
            
            window = GraphWindow(
                graph_storage=self.graph,
                filepath=self.graph_filepath,
                current_graph_llm=self.instruct_model,
                embedding_model=self.embedding_model
            )
            window.run()
        except Exception as e:
            ConsoleUI.print_error(f"Failed to launch visualization: {e}")
            Prompt.ask("Press Enter to continue", default="")
    
    def exit_app(self) -> bool:
        ConsoleUI.clear()
        ConsoleUI.print_header("Goodbye!", ColorTheme.PRIMARY)
        console.print("\n[italic]Thank you for using NIR NIR Chat System![/italic]\n")
        
        if self.graph and self.graph_filepath:
            self.graph.save(path=self.graph_filepath)
            ConsoleUI.print_success("Graph saved")
        
        return True


#----------------------------
#------main entry point------
#----------------------------

__all__ = ['ChatApplication']

def run_chat_app():
    app = ChatApplication()
    app.run()

if __name__ == "__main__":
    try:
        run_chat_app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        import traceback
        traceback.print_exc()