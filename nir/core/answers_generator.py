#All stuff related to answer generation is here


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel

from nir.prompts import answer_prompts
from nir.core.context_retriever import form_context_with_llm
from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.llm.manager import ModelManager
from nir.llm.providers import ModelConfig

plan_template = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_PLAN),
    ("human", "User request:\n{query}\n\n" + "Context:\n{context}\n\n")])

answer_template = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_FINAL_ANSWER),
    ("human", "User request:\n{query}\n\n" + "Narrative plan:\n{plan}\n\n")])

def generate_plan(query: str, context: str, llm: BaseLanguageModel) -> str:
    chain_plan = plan_template | llm
    result = chain_plan.invoke(({"query": query, "context": context}))
    return str(result)

def generate_answer_based_on_plan(query: str, plan: str, llm: BaseLanguageModel) -> str:
    chain_final = answer_template | llm
    result = chain_final.invoke(({"query": query, "plan": plan}))
    return str(result)

# manager = ModelManager()
# model_config = ModelConfig("llama3.2:latest")
# manager.create_chat_model("basic", "ollama", model_config)
# model = manager.get_chat_model("basic")

# graph_loaded = NetworkXGraph()
# graph_loaded.load("assets/graphs/graph_map_short.json")

# manager = ModelManager()
# embedding_model = manager.create_embedding_model(name="embeddings", option="ollama", model_name="nomic-embed-text:v1.5")
# model_config = ModelConfig(model_name="mistral:7b-instruct", temperature=0)
# manager.create_chat_model("graph_extraction", "ollama", model_config)
# llm = manager.get_chat_model("graph_extraction")

# query = "Create 10 idees for objects that can be found in Ariendale Village"
# context = form_context_with_llm(query, graph_loaded, llm, embedding_model, True)
# plan = generate_plan(query, context, model)
# answer = generate_answer_based_on_plan(query, plan, model)
# print(answer)