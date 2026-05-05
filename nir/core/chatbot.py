#--------------------------
#---------imports----------
#--------------------------

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.markdown import Markdown
from rich import box
from rich.align import Align

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from nir.core.answers_generator import generate_answer_based_on_plan, generate_plan
from nir.core.context_retriever import form_context_without_llm
from nir.data import loader
from nir.embedding.vector_store_loader import VectorStoreInfo
from nir.graph.graph_construction import create_embeddings, extract_graph, get_next_chunk_id, update_embeddings, update_graph
from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.llm.manager import ModelManager
from nir.llm.providers import ModelConfig

#-------------------------------
#---------Theme & Colors--------
#-------------------------------

console = Console()

class ColorTheme:
    PRIMARY = "#0fc23c"
    SECONDARY = "#04681d"
    ACCENT = "#ffaa00"
    DANGER = "#ff4444"
    INFO = "#00aeff"
    TEXT = "#D1D1D1"
    DARK = "#0a0a0a"

#-------------------------------
#---------Data Classes----------
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
        if self.temperature is not None:
            return f"{self.name} ({self.model_name}, {self.option}, temp: {self.temperature})"
        else:
            return f"{self.name} ({self.model_name}, {self.option})"

#-------------------------------
#---------Menu Components-------
#-------------------------------

class MenuItem:
    def __init__(self, title: str, action=None, data=None):
        self.title = title
        self.action = action
        self.data = data

class InteractiveMenu:
    def __init__(self, title: str, items: List[MenuItem], show_back: bool = True):
        self.title = title
        self.items = items
        self.selected_index = 0
        self.show_back = show_back
    
    def display(self) -> Optional[MenuItem]:
        while True:
            ConsoleUI.clear()
            ConsoleUI.print_header(self.title)
            
            console.print("\n[bold cyan]Use ↑/↓ to navigate, Enter to select, Esc to cancel[/bold cyan]\n")
            
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
            
            key = ConsoleUI.get_arrow_key()
            
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
        console.print(Panel(Align.center(text, style=color), 
                           border_style=ColorTheme.SECONDARY, 
                           box=box.HEAVY))
    @staticmethod
    def get_arrow_key():
        """Get arrow key input (cross-platform)"""
        import sys
        
        if sys.platform == 'win32':
            import msvcrt
            try:
                key = msvcrt.getch()
                if key == b'\xe0':  # Special keys prefix
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
                # Fallback to simple input if termios not available
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
        console.print(f"[green]✓[/green] {text}")
    
    @staticmethod
    def print_error(text: str):
        console.print(f"[red]✗[/red] {text}")
    
    @staticmethod
    def print_info(text: str):
        console.print(f"[cyan]ℹ[/cyan] {text}")
    
    @staticmethod
    def print_warning(text: str):
        console.print(f"[yellow]⚠[/yellow] {text}")
    
    @staticmethod
    def print_step(step: int, total: int, text: str):
        console.print(f"\n[bold {ColorTheme.ACCENT}][{step}/{total}][/bold {ColorTheme.ACCENT}] {text}")

#-------------------------------
#---------Form Components-------
#-------------------------------

class FormField:
    def __init__(self, label: str, default: Any = "", required: bool = True, field_type: str = "text",
                 options: List[str] = None):
        self.label = label
        self.default = default
        self.required = required
        self.field_type = field_type
        self.options = options
        self.value = default
        self.error = ""

class Form:
    def __init__(self, title: str, fields: List[FormField]):
        self.title = title
        self.fields = fields
        self.selected_field = 0
        self.editing = False
    
    def display(self) -> Optional[Dict[str, Any]]:
        while True:
            ConsoleUI.clear()
            ConsoleUI.print_header(self.title)
            
            # Display all fields
            console.print("\n[bold cyan]Press Enter to edit field, ↑/↓ to navigate[/bold cyan]\n")
            
            for idx, field in enumerate(self.fields):
                prefix = "→ " if idx == self.selected_field and not self.editing else "  "
                
                if field.field_type == "boolean":
                    value_display = "✓ Yes" if field.value else "✗ No"
                elif field.field_type == "select" and field.options:
                    value_display = f"[{field.value}]" if field.value else "[Not selected]"
                else:
                    value_display = field.value if field.value else "[empty]"
                
                required_mark = "[red]*[/red]" if field.required else "  "
                
                if field.error:
                    console.print(f"{prefix}[yellow]{field.label}:[/yellow] [red]{field.error}[/red]")
                else:
                    console.print(f"{prefix}{required_mark} [cyan]{field.label}:[/cyan] [green]{value_display}[/green]")
            
            if self.editing:
                console.print(f"\n[bold green]Editing: {self.fields[self.selected_field].label}[/bold green]")
                console.print("[dim]Type your value and press Enter. Esc to cancel editing.[/dim]")
            else:
                console.print(f"\n[dim]Enter: edit field | Esc: cancel | Ctrl+S: submit[/dim]")
            
            key = self.get_simple_key()
            
            if key == 'up' and not self.editing:
                self.selected_field = (self.selected_field - 1) % len(self.fields)
            elif key == 'down' and not self.editing:
                self.selected_field = (self.selected_field + 1) % len(self.fields)
            elif key == 'enter' and not self.editing:
                self.editing = True
                self.fields[self.selected_field].error = ""
            elif key == 'enter' and self.editing:
                self._submit_current_field()
            elif key == 'escape':
                if self.editing:
                    self.editing = False
                    self.fields[self.selected_field].value = self.fields[self.selected_field].default
                else:
                    return None
            elif key == 'ctrl+s' and not self.editing:
                if self._validate_form():
                    return self._get_results()
    
    def _submit_current_field(self):
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
    
    def _validate_form(self) -> bool:
        valid = True
        for field in self.fields:
            if field.required and not field.value:
                field.error = "This field is required"
                valid = False
        return valid
    
    def _get_results(self) -> Dict[str, Any]:
        import re
        result = {}
        for field in self.fields:
            # Нормализуем ключ: всё в нижний регистр, убираем спецсимволы
            key = field.label.lower()
            key = re.sub(r'[^a-z0-9_]', '_', key)  # заменяем всё кроме букв, цифр, _
            key = re.sub(r'_+', '_', key)         # убираем множественные подчеркивания
            key = key.strip('_')                  # удаляем подчеркивания по краям
            result[key] = field.value
        return result
    
    @staticmethod
    def get_simple_key():
        import sys
        if sys.platform == 'win32':
            import msvcrt
            key = msvcrt.getch()
            if key == b'\xe0':
                key = msvcrt.getch()
                if key == b'H':
                    return 'up'
                elif key == b'P':
                    return 'down'
            elif key == b'\r':
                return 'enter'
            elif key == b'\x1b':
                return 'escape'
            elif key == b'\x13':  # Ctrl+S
                return 'ctrl+s'
        else:
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
                elif ch == '\r':
                    return 'enter'
                elif ch == '\x1b':
                    return 'escape'
                elif ch == '\x13':
                    return 'ctrl+s'
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return None

#-------------------------------
#---------Chat Application------
#-------------------------------

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
    
    def run(self):
        self.intro()
        self.select_instruct_model()
        self.select_chat_model()
        self.select_or_create_graph()
        self.main_chat_loop()
    
    def intro(self):
        ConsoleUI.clear()
        ConsoleUI.print_header("NIR NIR INSTRUMENT", ColorTheme.PRIMARY)
        
        welcome_text = """
Welcome to the Graph-Aware Chat System!

This application allows you to:
• Extract knowledge graphs from documents
• Query the graph using natural language
• Update the graph based on conversations
• Visualize and edit the graph interactively

Let's set up your environment!
        """
        console.print(Panel(welcome_text, border_style=ColorTheme.SECONDARY, box=box.ROUNDED))
        Prompt.ask("\nPress Enter to continue", default="")
    
    def select_instruct_model(self):
        ConsoleUI.print_step(1, 5, "Selecting Graph Extraction Model")
        console.print("\n[italic]This model should be strict and precise. Use instruct models with temperature 0.0.[/italic]\n")       
        self.instruct_model_info = self.select_chat_model_from_manager("Graph Extraction Model")
        self.instruct_model = self.model_manager.get_chat_model(self.instruct_model_info.name)
        ConsoleUI.print_success(f"Model '{self.instruct_model_info.name}' loaded successfully")
    
    def select_chat_model(self):
        ConsoleUI.print_step(2, 5, "Selecting Chat Model")
        console.print("\n[italic]This model should be creative. Use temperature around 0.7.[/italic]\n")
        self.chat_model_info = self.select_chat_model_from_manager("Chat Model")
        self.chat_model = self.model_manager.get_chat_model(self.chat_model_info.name)
        ConsoleUI.print_success(f"Model '{self.chat_model_info.name}' loaded successfully")
    
    def select_chat_model_from_manager(self, model_type: str) -> ModelInfo:
        while True:
            models = self.model_manager.list_chat_models()
            if not models:
                ConsoleUI.print_warning("No models found. Let's create one!")
                Prompt.ask("\nPress Enter to continue", default="")
                self.create_new_chat_model()
                continue
            
            items = []
            for model in models:
                temp_str = f", temp: {model['temperature']}" if model['temperature'] else ""
                title = f"{model['name']} ({model['model_name']}, {model['option']}{temp_str})"
                items.append(MenuItem(title, data=model))
            
            items.append(MenuItem("Create new model", action="create"))
            
            menu = InteractiveMenu(f"Select {model_type}", items, show_back=True)
            selected = menu.display()
            
            if selected is None:
                continue
            
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

    def select_embedding_model(self) -> ModelInfo:
        while True:
            models = self.model_manager.list_embedding_models()
            
            if not models:
                ConsoleUI.print_warning("No embedding models found. Let's create one!")
                Prompt.ask("\nPress Enter to continue", default="")
                self.create_new_embedding_model()
                continue
            
            items = []
            for model in models:
                title = f"{model['name']} ({model['model_name']}, {model['option']})"
                items.append(MenuItem(title, data=model))
            
            items.append(MenuItem("Create new model", action="create"))
            
            menu = InteractiveMenu("Select Embedding Model", items, show_back=True)
            selected = menu.display()
            
            if selected is None:
                continue
            
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
            FormField("Model Name (user-friendly)", required=True),
            FormField("Model ID (official name)", required=True),
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
            model_name=result["model_id_official_name"],
            temperature=result["temperature"],
            max_tokens=result["max_tokens"]
        )
        
        api_key = result.get("api_key_if_needed")
        if api_key == "":
            api_key = None
        
        self.model_manager.create_chat_model(
            name=result["model_name_user_friendly"],
            option=result["option"],
            config=config,
            api_info=api_key
        )
        
        ConsoleUI.print_success(f"Model '{result['model_name_user_friendly']}' created successfully")
        Prompt.ask("\nPress Enter to continue", default="")
    
    def create_new_embedding_model(self):
        fields = [
            FormField("Model Name (user-friendly)", required=True),
            FormField("Model ID (official name)", required=True),
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
            name=result["model_name_user_friendly"],
            option=result["option"],
            model_name=result["model_id_official_name"],
            api_info=api_key
        )
        
        ConsoleUI.print_success(f"Embedding model '{result['model_name_user_friendly']}' created successfully")
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
        menu = InteractiveMenu("Select Graph", items, show_back=True)
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
        
        menu = InteractiveMenu("Graph Setup", items, show_back=False)
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
        self.embedding_model = self.model_manager.get_embedding_model(embedding_info.name)
        self.embedding_model_info = embedding_info

        fields = [
            FormField("Text file (from assets/documents)", required=True),
            FormField("Chunk Size", default="500", field_type="number"),
            FormField("Chunk Overlap", default="50", field_type="number"),
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
            Prompt.ask("Press Enter to continue", default="")
            return False

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("[cyan]Loading text...", total=1)
            if text_path.suffix == '.csv':
                data = loader.loadCSV_withColumns(path=str(text_path), columns=["text"])
            else:
                data = loader.loadTXT(str(text_path))
            progress.update(task, completed=1)
            
            task = progress.add_task("[cyan]Creating chunks...", total=1)
            chunks = loader.to_chunk_unique_id(
                docs=data,
                start_chunk_id=0,
                chunk_size=result["chunk_size"],
                chunk_overlap=result["chunk_overlap"]
            )
            progress.update(task, completed=1)
            
            task = progress.add_task("[cyan]Extracting graph...", total=len(chunks))
            
            self.graph = extract_graph(
                chunks=chunks,
                llm=self.instruct_model,
                embedding_model=self.embedding_model,
                graph_class=NetworkXGraph,
                preserve_all_data=False,
                language=result["language"]
            )
            
            progress.update(task, completed=len(chunks))
            
            task = progress.add_task("[cyan]Creating vector database...", total=1)
            vector_db_info = VectorStoreInfo(
                type="chromadb",
                info={
                    "name": f"{result['output_graph_name']}_db",
                    "path": "assets/databases/chroma_db"
                }
            )
            self.graph.create_vector_db(vector_db_info)
            create_embeddings(self.graph, self.graph.get_vector_db(), self.embedding_model)
            progress.update(task, completed=1)
        
        # Save graph
        output_path = f"assets/graphs/{result['output_graph_name']}.json"
        self.graph.save(path=output_path)
        self.graph_filepath = output_path
        
        ConsoleUI.print_success(f"Graph created and saved to {output_path}")
        Prompt.ask("\nPress Enter to continue", default="")
        return True
    
    def main_chat_loop(self):
        """Main application loop with menu and chat sessions"""
        ConsoleUI.print_step(4, 5, "Ready to Chat!")
        
        # Настройка истории (один раз при запуске)
        if not hasattr(self, 'history_configured'):
            self.add_history = Confirm.ask(
                "\n[bold]Use conversation history for context?[/bold]\n"
                "This will remember previous messages in the chat session", 
                default=True
            )
            self.history_configured = True
        
        while True:
            # Показываем главное меню
            ConsoleUI.clear()
            ConsoleUI.print_header("Graph-Aware Chat System")
            
            # Статусная информация
            status_text = f"""
    [bold {ColorTheme.PRIMARY}]Current Configuration:[/bold {ColorTheme.PRIMARY}]
    • Graph: [cyan]{Path(self.graph_filepath).name if self.graph_filepath else 'None'}[/cyan]
    • Chat Model: [cyan]{self.chat_model_info.display_string()}[/cyan]
    • Extraction Model: [cyan]{self.instruct_model_info.display_string()}[/cyan]
    • History: [cyan]{'Enabled' if self.add_history else 'Disabled'}[/cyan]
            """
            console.print(Panel(status_text, border_style=ColorTheme.SECONDARY, box=box.ROUNDED))
            
            # Меню опций
            console.print("\n[bold]Main Menu:[/bold]")
            console.print("  [cyan]1.[/cyan] [green]Start Chat Session[/green]")
            console.print("  [cyan]2.[/cyan] [yellow]Change Graph[/yellow]")
            console.print("  [cyan]3.[/cyan] [yellow]Change Chat Model[/yellow]")
            console.print("  [cyan]4.[/cyan] [yellow]Change Extraction Model[/yellow]")
            console.print("  [cyan]5.[/cyan] [yellow]Toggle Conversation History[/yellow]")
            console.print("  [cyan]6.[/cyan] [magenta]Visualize & Edit Graph[/magenta]")
            console.print("  [cyan]7.[/cyan] [red]Exit[/red]")
            
            choice = Prompt.ask("\nYour choice", choices=["1", "2", "3", "4", "5", "6", "7"], default="1")
            
            if choice == "1":
                # Запускаем чат-сессию
                self.run_chat_session()
                # После завершения чата возвращаемся в меню
                continue
            
            elif choice == "2":
                if self.select_or_create_graph():
                    ConsoleUI.print_success("Graph changed successfully!")
                    Prompt.ask("\nPress Enter to continue", default="")
            
            elif choice == "3":
                self.select_chat_model()
                ConsoleUI.print_success("Chat model changed successfully!")
                Prompt.ask("\nPress Enter to continue", default="")
            
            elif choice == "4":
                self.select_instruct_model()
                ConsoleUI.print_success("Extraction model changed successfully!")
                Prompt.ask("\nPress Enter to continue", default="")
            
            elif choice == "5":
                self.add_history = not self.add_history
                status = "enabled" if self.add_history else "disabled"
                ConsoleUI.print_success(f"Conversation history {status}")
                Prompt.ask("\nPress Enter to continue", default="")
            
            elif choice == "6":
                self.visualize_graph()
            
            elif choice == "7":
                if self.exit_app():
                    break

    def run_chat_session(self):
        """Run an interactive chat session with history"""
        ConsoleUI.clear()
        ConsoleUI.print_header("Interactive Chat Session")
        
        # Информация о сессии
        info_text = f"""
    [dim]Graph: {Path(self.graph_filepath).name}[/dim]
    [dim]Chat model: {self.chat_model_info.display_string()}[/dim]
    [dim]History: {'ON (conversation remembered)' if self.add_history else 'OFF (each query independent)'}[/dim]

    [bold green]Commands:[/bold green]
    • [cyan]/quit[/cyan] or [cyan]/q[/cyan] - Exit to main menu
    • [cyan]/clear[/cyan] - Clear current chat history
    • [cyan]/update[/cyan] - Update graph with last answer
    • [cyan]/history[/cyan] - Toggle conversation history on/off
    • [cyan]/help[/cyan] - Show this help
    • [cyan]/status[/cyan] - Show current session status
    
    [bold]Just type your question to chat with the graph![/bold]
    """
        console.print(Panel(info_text, border_style=ColorTheme.INFO, box=box.ROUNDED))
        
        # История текущей сессии
        chat_history = []
        last_answer = None
        session_add_history = self.add_history  # Сохраняем настройку на время сессии
        
        while True:
            console.print("\n[bold cyan]" + "─" * 50 + "[/bold cyan]")
            query = Prompt.ask("\n[bold green]You[/bold green]")
            
            # Обработка команд
            if query.lower() in ['/quit', '/q']:
                console.print("\n[yellow]Returning to main menu...[/yellow]")
                break
            
            elif query.lower() == '/clear':
                chat_history.clear()
                last_answer = None
                console.print("[green]✓ Chat history cleared![/green]")
                continue
            
            elif query.lower() == '/update':
                if last_answer:
                    self.update_graph_with_response(last_answer)
                else:
                    console.print("[yellow]⚠ No answer to update yet. Ask a question first![/yellow]")
                continue
            
            elif query.lower() == '/history':
                session_add_history = not session_add_history
                status = "ON" if session_add_history else "OFF"
                console.print(f"[green]✓ Conversation history turned {status}[/green]")
                continue
            
            elif query.lower() == '/status':
                status_text = f"""
    [bold]Session Status:[/bold]
    • Graph: {Path(self.graph_filepath).name}
    • Chat Model: {self.chat_model_info.name}
    • History: {'Enabled' if session_add_history else 'Disabled'}
    • Messages in history: {len(chat_history)}
    • Last answer ready for update: {'Yes' if last_answer else 'No'}
    """
                console.print(Panel(status_text, border_style=ColorTheme.INFO))
                continue
            
            elif query.lower() in ['/help', '/?']:
                console.print(Panel(info_text, title="Help", border_style=ColorTheme.INFO))
                continue
            
            # Обработка обычного вопроса
            with console.status("[bold cyan]🤔 Thinking...[/bold cyan]", spinner="dots"):
                try:
                    # Формируем контекст с историей если включено
                    if session_add_history and chat_history:
                        # Формируем контекст из истории
                        conversation_context = "\n".join([
                            f"User: {msg['question']}\nAssistant: {msg['answer']}"
                            for msg in chat_history[-3:]  # Последние 3 сообщения
                        ])
                        query_with_context = f"Previous conversation:\n{conversation_context}\n\nCurrent question: {query}"
                    else:
                        query_with_context = query
                    
                    context = form_context_without_llm(
                        query=query_with_context,
                        graph=self.graph,
                        embedding_model=self.embedding_model,
                        add_history=False  # Не используем глобальную историю
                    )
                    
                    # Генерируем план
                    plan = generate_plan(query, context, self.chat_model)
                    
                    # Генерируем ответ
                    answer = generate_answer_based_on_plan(query, plan, context, self.chat_model)
                    
                    last_answer = answer
                    
                    # Сохраняем в историю
                    if session_add_history:
                        chat_history.append({
                            "question": query,
                            "answer": answer
                        })
                    
                except Exception as e:
                    console.print(f"\n[red]✗ Error: {e}[/red]")
                    if not Confirm.ask("\nContinue chatting?", default=True):
                        break
                    continue
            
            # Показываем ответ
            console.print(f"\n[bold cyan]Assistant:[/bold cyan]")
            console.print(Panel(Markdown(answer), border_style=ColorTheme.PRIMARY, box=box.ROUNDED))
            
            # Краткая подсказка (не загромождаем)
            console.print("[dim](Type /help for commands, /quit to exit)[/dim]")
            
            # Не спрашиваем про обновление графа каждый раз, только если пользователь хочет
            # Но даем знать что можно обновить
            if last_answer:
                console.print("[dim]💡 Tip: Type /update to add this answer to the graph[/dim]")

    def chat_interaction(self):
        ConsoleUI.clear()
        ConsoleUI.print_header("Chat with Your Graph")
        
        console.print(f"[dim]Working with graph: {Path(self.graph_filepath).name}[/dim]\n")
        
        query = Prompt.ask("[bold green]Your question[/bold green]")
        
        if query.lower() in ['q', 'quit', 'exit']:
            return
        
        # Используем Live для динамического обновления вместо Progress
        from rich.live import Live
        from rich.text import Text
        
        status_text = Text("Initializing...", style="cyan")
        
        with Live(status_text, refresh_per_second=10, transient=True) as live:
            try:
                status_text = Text("🔍 Searching graph...", style="cyan")
                live.update(status_text)
                
                context = form_context_without_llm(
                    query=query,
                    graph=self.graph,
                    embedding_model=self.embedding_model,
                    add_history=self.add_history
                )
                
                status_text = Text("🧠 Generating plan...", style="cyan")
                live.update(status_text)
                
                plan = generate_plan(query, context, self.chat_model)
                
                status_text = Text("✍️ Writing answer...", style="cyan")
                live.update(status_text)
                
                answer = generate_answer_based_on_plan(query, plan, context, self.chat_model)
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                Prompt.ask("\nPress Enter to continue", default="")
                return
        
        ConsoleUI.clear()
        ConsoleUI.print_header("Answer")
        console.print(Panel(Markdown(answer), border_style=ColorTheme.PRIMARY, box=box.ROUNDED))
        
        if Confirm.ask("\n[bold yellow]Update graph with this information?[/bold yellow]", default=False):
            self.update_graph_with_response(answer)
        
        Prompt.ask("\nPress Enter to continue", default="")
    
    def update_graph_with_response(self, response: str):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("[cyan]Updating graph...", total=None)
            
            greatest_id = get_next_chunk_id(self.graph)
            chunks = loader.to_chunk_unique_id(
                docs=response,
                start_chunk_id=greatest_id
            )
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
        console.print("\n[italic]Thank you for using Graph-Aware Chat System![/italic]\n")
        
        if self.graph and self.graph_filepath:
            self.graph.save(path=self.graph_filepath)
            ConsoleUI.print_success("Graph saved")
        
        return True

#-------------------------------
#---------Main Entry Point------
#-------------------------------

__all__ = ['ChatApplication']

def run_chat_app():
    """Function to run the chat application from external code"""
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