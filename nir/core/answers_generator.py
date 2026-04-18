#All stuff related to answer generation is here



from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel

from nir.prompts import answer_prompts


import re

def parse_llm_answer(text: str, content_type: str) -> str:
    print("--MODEL ANSWER--")
    print(text)
    if not text or not text.strip():
        return ""
    text = text.strip()
    text = re.sub(r'^```(?:\w+)?\s*\n?', '', text)
    if text.rstrip().endswith('```'):
        text = text.rstrip()[:-3]
    text = text.strip()

    tag_pattern = rf'<{content_type}\s*>(.*?)</\s*{content_type}\s*>'
    match = re.search(tag_pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    text = re.sub(r'<reasoning\s*>.*?</\s*reasoning\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(rf'<{content_type}\s*>|</\s*{content_type}\s*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<reasoning\s*>|</\s*reasoning\s*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


basic_plan_template_en = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_PLAN_BASIC_EN),
    ("human", "User request:\n{query}\n\n" + "Context:\n{context}\n\n")])

theoretical_plan_template_en = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_PLAN_WITH_THEORY_EN),
    ("human", "User request:\n{query}\n\n" + "Context:\n{context}\n\n")])

context_filtering_template_en = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_CONTEXT_FILTRATION_EN),
    ("human", "User request:\n{query}\n\n" + "Context:\n{context}\n\n")])



answer_template_based_on_plan_en = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_FINAL_ANSWER_BASED_ON_PLAN_EN),
    ("human", "User request:\n{query}\n\n" + "Narrative plan:\n{plan}\n\n" + "Context: \n{context}\n\n")])

answer_template_based_on_context_en = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_FINAL_ANSWER_BASED_ON_CONTEXT_EN),
    ("human", "User request:\n{query}\n\n" + "Context: \n{context}\n\n")])



# for russian language
basic_plan_template_ru = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_PLAN_BASIC_RU),
    ("human", "User request:\n{query}\n\n" + "Context:\n{context}\n\n")])

theoretical_plan_template_ru = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_PLAN_WITH_THEORY_RU),
    ("human", "User request:\n{query}\n\n" + "Context:\n{context}\n\n")])

context_filtering_template_ru = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_CONTEXT_FILTRATION_RU),
    ("human", "User request:\n{query}\n\n" + "Context:\n{context}\n\n")])



answer_template_based_on_plan_ru = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_FINAL_ANSWER_BASED_ON_PLAN_RU),
    ("human", "User request:\n{query}\n\n" + "Narrative plan:\n{plan}\n\n" + "Context: \n{context}\n\n")])

answer_template_based_on_context_ru = ChatPromptTemplate.from_messages([
    ("system", answer_prompts.SYSTEM_PROMPT_FINAL_ANSWER_BASED_ON_CONTEXT_RU),
    ("human", "User request:\n{query}\n\n" + "Context: \n{context}\n\n")])


def generate_plan(query: str, context: str, llm: BaseLanguageModel, include_theory: bool = True, language: str = "en") -> str:
    if include_theory:
        if language == "ru":
            chain_plan = theoretical_plan_template_ru | llm
        else:
            chain_plan = theoretical_plan_template_en | llm
    else:
        if language == "ru":
            chain_plan = basic_plan_template_ru | llm
        else:
            chain_plan = basic_plan_template_en | llm
    result = chain_plan.invoke(({"query": query, "context": context}))
    result = parse_llm_answer(result, "plan")
    return str(result)

def filter_context(query: str, context: str, llm: BaseLanguageModel, language: str = "en") -> str:
    if language == "ru":
        chain_context = context_filtering_template_ru | llm
    else:
        chain_context = context_filtering_template_en | llm
    result = chain_context.invoke(({"query": query, "context": context}))
    result = parse_llm_answer(result, "filtered_context")
    return str(result)

def generate_answer_based_on_plan(query: str, plan: str, context: str, llm: BaseLanguageModel, language: str = "en") -> str:
    if language == "ru":
        chain_final = answer_template_based_on_plan_ru | llm
    else:
        chain_final = answer_template_based_on_plan_en | llm
    result = chain_final.invoke(({"query": query, "plan": plan, "context": context}))
    return str(result)

def generate_answer_based_on_context(query: str, context: str, llm: BaseLanguageModel, language: str = "en") -> str:
    if language == "ru":
        chain_final = answer_template_based_on_context_ru | llm
    else:
        chain_final = answer_template_based_on_context_en | llm
    result = chain_final.invoke(({"query": query, "context": context}))
    result = parse_llm_answer(result, "answer")
    return str(result)