#All stuff related to answer generation is here



from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel

from nir.prompts import answer_prompts


import re
from typing import Optional

def parse_llm_answer(text: str, content_type: str) -> str:
    text = text.strip()
    if text.startswith('```'):
        first_nl = text.find('\n')
        text = text[first_nl+1:] if first_nl != -1 else text
    if text.endswith('```'):
        text = text[:-3].rstrip()
    reasoning_match = re.search(
        r'<reasoning\s*>(.*?)</\s*reasoning\s*>', 
        text, re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        print(reasoning_match.group(1).strip())

    answer_match = re.search(
        rf'<{content_type}\s*>(.*?)</\s*{content_type}\s*>', 
        text, re.DOTALL | re.IGNORECASE
    )
    return answer_match.group(1).strip()


basic_plan_template = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_PLAN_BASIC),
    ("human", "User request:\n{query}\n\n" + "Context:\n{context}\n\n")])

theoretical_plan_template = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_PLAN_WITH_THEORY),
    ("human", "User request:\n{query}\n\n" + "Context:\n{context}\n\n")])

context_filtering_template = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_CONTEXT_FILTRATION),
    ("human", "User request:\n{query}\n\n" + "Context:\n{context}\n\n")])



answer_template_based_on_plan = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_FINAL_ANSWER_BASED_ON_PLAN),
    ("human", "User request:\n{query}\n\n" + "Narrative plan:\n{plan}\n\n" + "Context: \n{context}\n\n")])

answer_template_based_on_context = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_FINAL_ANSWER_BASED_ON_CONTEXT),
    ("human", "User request:\n{query}\n\n" + "Context: \n{context}\n\n")])



def generate_plan(query: str, context: str, llm: BaseLanguageModel, include_theory: bool = True) -> str:
    if include_theory:
        chain_plan = theoretical_plan_template | llm
    else:
        chain_plan = basic_plan_template | llm
    result = chain_plan.invoke(({"query": query, "context": context}))
    result = parse_llm_answer(result, "plan")
    print(result)
    return str(result)

def filter_context(query: str, context: str, llm: BaseLanguageModel) -> str:
    chain_context = context_filtering_template | llm
    result = chain_context.invoke(({"query": query, "context": context}))
    result = parse_llm_answer(result, "filtered_context")
    print(result)
    return str(result)

def generate_answer_based_on_plan(query: str, plan: str, context: str, llm: BaseLanguageModel) -> str:
    chain_final = answer_template_based_on_plan | llm
    result = chain_final.invoke(({"query": query, "plan": plan, "context": context}))
    return str(result)

def generate_answer_based_on_context(query: str, context: str, llm: BaseLanguageModel) -> str:
    chain_final = answer_template_based_on_context | llm
    result = chain_final.invoke(({"query": query, "context": context}))
    result = parse_llm_answer(result, "answer")
    return str(result)