#This is working chat

from nir.data import loader

# manager = ModelManager()
# model_config = ModelConfig("llama3.2:latest")
# manager.create_chat_model("basic", "ollama", model_config)
# manager.create_embedding_model("basic", "ollama", "nomic-embed-text:v1.5")

# template = """
#     You are a professional narrative designer, working in game industry. Answer a question.
#     Here is some relevant information: {reviews}
#     Here is the question to answer: {question}
# """

# model = manager.get_chat_model("basic")
# embeddings = manager.get_embedding_model("basic")

# prompt = ChatPromptTemplate.from_template(template)
# chain = prompt | model

# graph = nx.read_graphml("assets/graphs/graph.graphml")
# retriever = knowledge_graph.create_embeddings(list(graph.nodes), embeddings)

# while True:
#     print("\n---------------------")
#     question = input("Ask your question (q to quit): ")
#     print("\n")
#     if question == "q":
#         break

#     precontext = context_retriever.get_context_networkX(question, retriever, graph)
#     context = context_retriever.prepare_context(precontext)
#     result = chain.invoke({"reviews": context, "question": question})
#     print(result)

string = """
This is a huge string for checking wxtraction of list of documents. 
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""
data = loader.convertFromString(string)
print(data)