import re
from typing import List, Tuple
from collections import Counter

import mauve
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate


def compute_mauve(generated: str, reference: str, model_id: str = "gpt2") -> float:
    if not generated.strip() or not reference.strip():
        return 0.0

    out = mauve.compute_mauve(
        p_text=[reference],
        q_text=[generated],
        device_id=0,
        verbose=False,
    )
    return float(out.mauve)


def _get_ngrams(text: str, n: int) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def compute_distinct_n(generated: str, n: int = 2) -> float:
    ngrams = _get_ngrams(generated, n)
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)

def compute_repetition_n(generated: str, n: int = 2) -> float:
    ngrams = _get_ngrams(generated, n)
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def compute_world_consistency(
    original_context: str,
    generated_text: str,
    llm: BaseLanguageModel
) -> float:
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert game designer evaluating narrative consistency."),
        ("human", (
            "Original game world description:\n{original_context}\n\n"
            "Generated text:\n{generated_text}\n\n"
            "On a scale from 0 to 1, how well does the generated text fit into the same game world? "
            "Consider lore, tone, entities, rules, and style. "
            "Respond ONLY with a number between 0.0 and 1.0, with one decimal place (e.g., 0.7)."
        ))
    ])

    chain = prompt_template | llm
    try:
        response = chain.invoke({
            "original_context": original_context,
            "generated_text": generated_text
        })
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        match = re.search(r"(\d*\.?\d+)", answer)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)
        else:
            return 0.0
    except Exception as e:
        print(f"Error in world consistency evaluation: {e}")
        return 0.0