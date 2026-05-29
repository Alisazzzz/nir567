# Narrative Generation with Knowledge Graph-Based RAG


This project is a prototype system for generating text-based narrative elements such as character descriptions, quests, dialogues, lore fragments, and other narrative content, primarily for video games.

Inspired by graph-based RAG systems such as GraphRAG, LightRAG, and PathRAG, the project is also based on several research related to game narrative or gameplay generation using structured world representations that are used for context. The main goal of the method is to improve the consistency between generated narrative elements and the existing game world compared to standard RAG approaches.

In this method, the game world is represented as a knowledge graph consisting of six types of nodes: locations, environmental elements, characters, character groups, items, and events, as well as edges connecting them. All nodes except event nodes can contain multiple states representing changes to an entity caused by events. Edges also contain temporal information describing when a relationship appears and when it disappears. If an edge has no timestamps, it is considered to exist throughout the entire history of the world. As a result, the graph represents not only the structure of the game world and relationships between entities, but also the evolution of the world over time and the impact of events on its state.

The method is implemented as a console application that allows users to experiment with the complete extraction and generation pipeline.

## Contents
- [Method Overview](#method-overview)
- [Installation](#installation)

## Method overview
This repository implements a pipeline that begins with knowledge graph extraction and proceeds to narrative element generation based on context retrieved from the graph. The system also supports updating the graph with newly generated content.

The knowledge graph extraction pipeline uses a language model to extract entities, relationships, and event impacts from unstructured text describing the game world. Such text can include scripts, design documents, lore descriptions, or other narrative materials.

The extraction pipeline consists of the following stages:

1. *(Optional, but default for console app right now)* Future node names are extracted from text chunks and merged to improve entity consistency.
2. Nodes and edges with a predefined structure are extracted from text chunks and merged into a graph.
3. Event impacts are extracted for every event and for every node or edge influenced by that event. These impacts are then applied to the graph, creating temporal states for nodes and temporal existence intervals for edges.

The narrative generation pipeline varies depending on the selected mode.

The default mode is a two-stage generation process:
1. The language model first generates a high-level plan for the future response, including narrative direction, emotional tone, conflicts, and important contextual details.
2. The language model then generates the final narrative element based on the generated plan and the retrieved graph context.

The system also supports a one-stage generation mode, in which the language model directly generates the final response based only on the raw context retrieved from the graph.

Context retrieval combines semantic similarity search with graph traversal. The system retrieves relevant entities, neighboring nodes, and paths between related entities while also taking temporal constraints into account. This allows the generated content to remain more logically and chronologically consistent with the current state of the game world.

Additionally, since the system is implemented as a chatbot-based tool, conversation history is also included in the context provided to the language model.

## Installation
This project was developed and tested only on Windows environments.

The system also requires a CUDA-compatible GPU for efficient execution of language and embedding models. CPU-only execution is currently not officially supported and may result in significantly reduced performance or unsupported behavior.

To use method and instruments provided in this repository, follow this steps:
#### 0. Install Python
This project requires Python 3.13. The codebase was tested with Python 3.13 only, and compatibility with other Python versions is currently unknown.
#### 1. Create a folder for the project
Create any folder where you want to store the repository.
#### 2. Clone the repository
Open the folder in a terminal and run:
```bash
git clone https://github.com/Alisazzzz/nir567.git
```
You can also download the repository manually using any other preferred method.
#### 3. Run .bat file
Go inside nir567 folder and launch install.bat file. This will create virtual environment and install all dependencies needed for repository.

#### 4. Activate the virtual environment
After all dependencies have been installed, you can close window from .bat file and open command line from nir567 folder. In this command line, activate virtual environment.
```bash
.venv/Scripts/activate
```
If you get an execution policy error, run:
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```
Then activate the environment again.
### Running the Project
After activating virtual environment, run:
```bash
python main.py
```

Detailed descriptions of the next steps can be found in english in [project wiki](https://github.com/Alisazzzz/nir567/wiki/NIR567-WIKI).
OR for russian you can find more detailed instuctions [here](https://app.notion.com/p/3611b883cb478010b227eed4a9eb4ba7?source=copy_link)
