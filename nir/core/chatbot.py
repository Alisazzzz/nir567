#This is working chat

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import networkx as nx

from langchain_ollama import OllamaEmbeddings

from nir.graph import knowledge_graph
from nir.core import context_searcher

model = OllamaLLM(model="llama3.2:latest")
embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

template = """
    You are a professional narrative designer, working in game industry. Answer a question.
    Here is some relevant information: {reviews}
    Here is the question to answer: {question}
"""

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