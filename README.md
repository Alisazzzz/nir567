# Narrative Generation with Knowledge Graph-Based RAG



Hello there! This is a repo for a prototype system, that is created for generation video games narrative elements using stuff like Language Models (local or remote, as you wish), Retrieval-Augmented Generation (RAG), an a custom knowledge graph that represents the game world.
The system is based on theoretical research, main points of which are:
  1. Structured representations such as knowledge graphs can improve the consistency between the game world and newly generated narrative elements more effectively than standard RAG systems.
  2. No other points, I'm sorry, only first one.

The project is designed to assist narrative designers and game developers by automating the creation of text-based narrative content such as character descriptions, location descriptions, quests, dialogues and so on. 

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
