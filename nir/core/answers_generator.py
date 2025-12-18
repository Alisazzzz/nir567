#All stuff related to answer generation is here


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel

from nir.prompts import answer_prompts

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