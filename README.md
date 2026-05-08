# Narrative Generation with Knowledge Graph-Based RAG


This project is a prototype system for generating text-based narrative elements such as character descriptions, quests, dialogues, lore fragments, and other narrative content, primarily for video games.

Inspired by graph-based RAG systems such as GraphRAG, LightRAG, and PathRAG, the project is also based on several research related to game narrative or gameplay generation using structured world representations that are used for context. The main goal of the method is to improve the consistency between generated narrative elements and the existing game world compared to standard RAG approaches.

In this method, the game world is represented as a knowledge graph consisting of six types of nodes: locations, environmental elements, characters, character groups, items, and events, as well as edges connecting them. All nodes except event nodes can contain multiple states representing changes to an entity caused by events. Edges also contain temporal information describing when a relationship appears and when it disappears. If an edge has no timestamps, it is considered to exist throughout the entire history of the world. As a result, the graph represents not only the structure of the game world and relationships between entities, but also the evolution of the world over time and the impact of events on its state.

The method is implemented as a console application that allows users to experiment with the complete extraction and generation pipeline.

## Requirements
Python 3.13  
  > The project was tested with Python 3.13. Compatibility with other Python versions has not been verified.

## Installation

#### 1. Create a folder for the project
Create any folder where you want to store the repository.
#### 2. Clone the repository
Open the folder in a terminal and run:
```bash
git clone https://github.com/Alisazzzz/nir567.git
```
You can also download the repository manually using any other preferred method.
#### 3. Navigate to the project folder
```bash
cd nir567
```
#### 4. (Recommended) Create a virtual environment
```bash
py -3.13 -m venv .venv
```
#### 5. Activate the virtual environment
```bash
.venv/Scripts/activate
```
If you get an execution policy error, run:
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```
Then activate the environment again.
#### 6. Install required dependencies
```bash
pip install -r requirements.txt
```
#### 7. Install spaCy language models
```bash
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm
```
### Running the Project
After everything is installed, run:
```bash
python main.py
```
