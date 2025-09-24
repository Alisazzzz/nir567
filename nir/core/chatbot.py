#This is working chat

from langchain_core.prompts import ChatPromptTemplate
import networkx as nx

from nir.graph import knowledge_graph
from nir.core import context_searcher

from nir.llm.providers import ModelConfig
from nir.llm.manager import ModelManager

manager = ModelManager()
model_config = ModelConfig("llama3.2:latest")
manager.create_chat_model("basic", "ollama", model_config)
manager.create_embedding_model("basic", "ollama", "nomic-embed-text:v1.5")

template = """
    You are a professional narrative designer, working in game industry. Answer a question.
    Here is some relevant information: {reviews}
    Here is the question to answer: {question}
"""

model = manager.get_chat_model("basic")
embeddings = manager.get_embedding_model("basic")

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

graph = nx.read_graphml("assets/graphs/graph.graphml")
retriever = knowledge_graph.create_embeddings(list(graph.nodes), embeddings)

while True:
    print("\n---------------------")
    question = input("Ask your question (q to quit): ")
    print("\n")
    if question == "q":
        break

    precontext = context_searcher.get_context_networkX(question, retriever, graph)
    context = context_searcher.prepare_context(precontext)
    result = chain.invoke({"reviews": context, "question": question})
    print(result)