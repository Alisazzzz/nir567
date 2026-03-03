#All stuff about server for web app is here




#--------------------------
#---------imports----------
#--------------------------

import json
import os
from fastapi import APIRouter
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from nir.core.answers_generator import generate_answer_based_on_plan, generate_plan
from nir.core.chat_history import ChatHistory
from nir.core.context_retriever import form_context_with_llm, form_context_without_llm
from nir.data import loader
from nir.embedding.vector_store_loader import VectorStoreInfo
from nir.graph.graph_construction import create_embeddings, extract_graph, get_next_chunk_id, update_embeddings
from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.llm.manager import ModelManager
from nir.llm.providers import ModelConfig
import webapp.api.response_structures as models
import webapp.api.server_answers_en as answers



model_manager = ModelManager()

#----------------------------------------------------
#---------variables while server is working----------
#----------------------------------------------------

router = APIRouter(prefix="/api")
graphs_folder_path = "./assets/graphs"
history_folder_path = "./assets/chats"
databases_folder_path = "./assets/databases/chroma_db"

current_graph = NetworkXGraph()
current_graph_path = "" #filename.json

current_chat_history = ChatHistory("")

current_embedding_model: Embeddings
current_chat_model: BaseLanguageModel
current_chat_model_name = "" #name
current_instruct_model: BaseLanguageModel
current_instruct_model_name = "" #name

current_max_tokens = 0

#--------------------------
#-----additional stuff-----
#--------------------------

def find_chat_history_by_graph(graph_path: str) -> Optional[str]:
    if not os.path.exists(history_folder_path):
        return None
    graph_path_normalized = os.path.normpath(graph_path)

    for filename in os.listdir(history_folder_path):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(history_folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            stored_graph_path = data.get("graph_path")
            if stored_graph_path:
                stored_graph_path_normalized = os.path.normpath(stored_graph_path)
                if stored_graph_path_normalized == graph_path_normalized:
                    return file_path                
        except (json.JSONDecodeError, KeyError, IOError):
            continue
    return None



#---------------------------------
#---------work with chat----------
#---------------------------------

@router.post("/chat", response_model=models.ChatResponse)
def chat(message: models.ChatMessage):
    if current_graph_path == "":
        return models.ChatResponse(answer=answers.NO_GRAPH_CHAT_ANSWER, model=current_chat_model_name if current_chat_model_name != "" else answers.NO_MODEL_MODEL_TITLE)
    if current_chat_model_name == "":
        return models.ChatResponse(answer=answers.NO_MODEL_CHAT_ANSWER, model=answers.NO_MODEL_MODEL_TITLE)
    
    query = message.text
    context = ""
    if message.use_timestamps:
        try:
            context = form_context_with_llm(
                query=query, 
                graph=current_graph, 
                llm=current_chat_model,
                embedding_model=current_embedding_model, 
                add_history=message.add_history
            )
        except:
            return models.ChatResponse(answer=answers.ERROR_RETRIEVING_CONTEXT_ANSWER, model=current_chat_model_name)
    else:
        context = form_context_without_llm(
            query=query, 
            graph=current_graph, 
            embedding_model=current_embedding_model, 
            add_history=message.add_history
        )
    try:
        answer_plan = generate_plan(query, context, current_chat_model)
        answer_final = generate_answer_based_on_plan(query, answer_plan, context, current_chat_model)
    except:
        return models.ChatResponse(answer=answers.ERROR_GENERATING_ANSWER_ANSWER, model=current_chat_model_name)

    current_chat_history.add_message_to_history("user", query)
    current_chat_history.add_message_to_history("assistant", answer_final)
    if current_chat_history.file_path:
        current_chat_history.save()

    return models.ChatResponse(answer=answer_final, model=current_chat_model_name)



#---------------------------------------------------
#---------work with chat or instruct model----------
#---------------------------------------------------

@router.get("/models/get-current", response_model=models.SelectedModel)
def get_current_model(request: models.ChatOrInstruct):
    if (request.model_type == "chat"):
        if not current_chat_model_name == "":
            return models.SelectedModel(filename=current_chat_model_name, model_type="chat")
        else:
            return models.SelectedModel(filename=answers.NO_CHAT_MODEL_SELECTED, model_type="chat")
    else:
        if not current_instruct_model_name == "":
            return models.SelectedModel(filename=current_instruct_model_name, model_type="instruct")
        else:
            return models.SelectedModel(filename=answers.NO_INSTRUCT_MODEL_SELECTED, model_type="instruct")
    
  
@router.post("/models/load-all", response_model=List[models.ExistingModel])
def load_models(request: models.ChatOrInstruct):
    all_models = model_manager.list_chat_models()
    result = []
    for model in all_models:
        existing_model = models.ExistingModel(
            name=model.get("name"), 
            option=model.get("option"), 
            model_name=model.get("model_name"), 
            max_tokens=model.get("max_tokens"), 
            temperature=model.get("temperature"), 
            is_current=(model.get("name") == current_chat_model_name if request.model_type == "chat" else model.get("name") == current_instruct_model_name)
        )
        result.append(existing_model)
    return result


@router.post("/models/select", response_model=models.SelectedModel)
def select_model(model: models.SelectedModel):
    if model.model_type == "chat":
        global current_chat_model_name, current_chat_model, current_max_tokens
        current_chat_model_name = model.name
        current_chat_model = model_manager.get_chat_model(model.name)
        current_max_tokens = model_manager.get_max_tokens_for_model(model.name)
    elif model.model_type == "instruct":
        global current_instruct_model_name, current_instruct_model
        current_instruct_model_name = model.name
        current_instruct_model = model_manager.get_chat_model(model.name)
    return models.SelectedModel(model.name, model.model_type)


@router.post("/models/create-and-select", response_model=models.SelectedModel)
def create_and_select_model(model_info: models.ModelToCreate):
    model_config = ModelConfig(
        model_name=model_info.model_name,
        temperature=model_info.temperature,
        max_tokens=model_info.max_tokens
    )
    created_model = model_manager.create_chat_model(
            name=model_info.name,
            option=model_info.option,
            config=model_config,
            api_info=model_info.api,
    )
    if model_info.model_type == "chat":
        global current_chat_model, current_chat_model_name
        current_chat_model = created_model
        current_chat_model_name=model_info.name
        return models.SelectedModel(name=current_chat_model_name, model_type=model_info.model_type)
    else:
        global current_instruct_model, current_instruct_model_name
        current_instruct_model = created_model
        current_instruct_model_name=model_info.name
        return models.SelectedModel(name=current_instruct_model_name, model_type=model_info.model_type)



#-----------------------------------
#---------work with graphs----------
#-----------------------------------

@router.post("/graph/update", response_model=models.ExistingGraph)
def update_graph(text: models.ChatMessage):
    greatest_id = get_next_chunk_id(current_graph)
    chunks = loader.to_chunk_unique_id(docs=text.text, start_chunk_id=greatest_id)
    update_graph(chunks, current_instruct_model, current_embedding_model, current_graph)
    current_graph.save(current_graph_path)
    update_embeddings(current_graph, current_graph.get_vector_db(), current_embedding_model)
    return models.ExistingGraph(filename=current_graph_path, document=current_graph.get_document_filename(), is_current=True)


@router.get("/graph/get-current", response_model=models.ExistingGraph)
def get_current_graph():
    if not current_graph_path == "":
        return models.ExistingGraph(filename=current_graph_path, document=current_graph.get_document_filename(), is_current=True)
    else:
        return models.ExistingGraph(filename=answers.NO_GRAPH_SELECTED, document=answers.NO_DOCUMENT_LOADED, is_current=False)


# select graph modal window
@router.get("/graph/load-all", response_model=List[models.ExistingGraph])
def load_graphs():
    all_entries = os.listdir(graphs_folder_path)
    filenames = [entry for entry in all_entries if os.path.isfile(os.path.join(graphs_folder_path, entry))]
    graphs = []
    for name in filenames:
        graph = models.ExistingGraph(filename=name, document=current_graph.get_document_filename(), is_current=(name == current_graph_path))
        graphs.append(graph)
    return graphs


@router.post("/graph/select", response_model=Dict[str, Any])
def select_graph(message: models.SelectedGraph):
    global current_graph_path, current_chat_history
    current_graph.load(filepath=os.path.join(graphs_folder_path, message.filepath))
    current_graph_path = message.filepath
    chat_filepath = find_chat_history_by_graph(os.path.join(graphs_folder_path, current_graph_path))
    history = []
    if chat_filepath != None: 
        current_chat_history = ChatHistory.load(chat_filepath)
        history = current_chat_history.messages
    return { 
        "existing_graph" : models.ExistingGraph(filename=message.filepath, document=current_graph.get_document_filename(), is_current=True),
        "chat_history" : models.ChatHistory(history)
    }


# create graph modal window
@router.post("/graph/create-and-select", response_model=models.ExistingGraph)
def create_graph(graph_info: models.GraphInfo):
    global current_graph, current_graph_path, current_chat_history
    data = loader.loadTXT(path=graph_info.document_filepath)
    chunks = loader.to_chunk_unique_id(docs=data, start_chunk_id=0)

    current_embedding_model = model_manager.get_embedding_model(graph_info.embedding_model_name)
    try:
        graph = extract_graph(chunks=chunks, llm=current_instruct_model, embedding_model=current_embedding_model, graph_class=NetworkXGraph)
    except:
        return models.ExistingGraph(filename=answers.ERROR_CREATING_GRAPH, document=graph_info.document_filepath, is_current=False)
    vector_db_info = VectorStoreInfo(
        type="chromadb",
        info={ 
            "name" : graph_info.graph_filename,
            "path" : databases_folder_path
        }
    )
    graph.create_vector_db(vector_db_info)
    create_embeddings(graph, graph.get_vector_db(), current_embedding_model)

    filename = graph_info.graph_filename + ".json"
    graph.save(filepath=os.path.join(graphs_folder_path, filename))
    history_filename = graph_info.graph_filename + "_chat_history.json"
    current_chat_history = ChatHistory(
        graph_path=os.path.join(graphs_folder_path, filename), 
        file_path = os.path.join(history_folder_path, history_filename)
    )
    current_graph = graph
    current_graph_path = graph_info.graph_filename + ".json"
    return models.ExistingGraph(filename=current_graph_path, document=graph_info.document_filepath, is_current=True)


# embedding model modal window
@router.get("/embedding/load-all", response_model=models.ExistingEmbedding)
def load_all_embeddings():
    embeddings_models = model_manager.list_embedding_models()
    result = []
    for model in embeddings_models:
        existing_embedding = models.ExistingEmbedding(
            name=model.get("name"),
            model_name=model.get("model_name"),
            option=model.get("option")
        )
        result.append(existing_embedding)
    return result


@router.post("/embedding/select", response_model=models.ExistingEmbedding)
def select_embedding(selected: models.SelectedEmbedding):
    global current_embedding_model
    current_embedding_model = model_manager.get_embedding_model(selected.name)
    return models.ExistingEmbedding(name=selected.name, option=selected.model_type)


@router.post("/embedding/create-and-select", response_model=models.ExistingEmbedding)
def create_embedding_model(embeddings_info: models.EmbeddingsInfo):
    global current_embedding_model
    current_embedding_model = model_manager.create_embedding_model(
        name=embeddings_info.name,
        option=embeddings_info.option,
        model_name=embeddings_info.model_name,
        api_info=embeddings_info.api_info
    )
    return models.ExistingEmbedding(name=embeddings_info.name, option=embeddings_info.option)