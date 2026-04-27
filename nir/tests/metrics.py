#All stuff for certain metrics is here



#--------------------------
#---------imports----------
#--------------------------

import re
from typing import Dict, List, Optional
from collections import Counter, defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from datasets import Dataset
from bert_score import score as bert_score_fn
import os
import mauve

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from nir.graph.graph_structures import Edge
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.prompts import testing_prompts

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import RunConfig, evaluate
from ragas.metrics import (
    answer_similarity,
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness, 
)



#--------------------------
#-----additional stuff-----
#--------------------------

def parse_llm_answer(text: str, content_type: str) -> str:
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

def get_ngrams(text: str, n: int) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def split_context_by_lines(context: str) -> list[str]:
    if not context:
        return []
    return context.split('\n')



#--------------------------
#-----prompt templates-----
#--------------------------

prompt_template_interestingness_en = ChatPromptTemplate.from_messages([
    ("system", testing_prompts.SYSTEM_PROMPT_INTERESTINGNESS_EN),
    ("human", (
        "Task description:\n{task_description}\n\n"
        "Newly created text:\n{generated_text}\n\n" ))
])

prompt_template_world_consistency_en = ChatPromptTemplate.from_messages([
    ("system", testing_prompts.SYSTEM_PROMPT_WORLD_CONSISTENCY_EN),
    ("human", (
        "Original game world description:\n{original_context}\n\n"
        "Newly created text:\n{generated_text}\n\n" ))
])



#----------------------------------------------------
#---------metrics for text quality analysis----------
#----------------------------------------------------

def compute_mauve(generateds: list[str], references: list[str], model_id: str = "gpt2") -> float:
    out = mauve.compute_mauve(p_text=references, q_text=generateds, device_id=0, mauve_scaling_factor=2)
    return float(out.mauve)


def compute_self_bleu(generateds: list[str], max_n: int=4) -> float:
    if len(generateds) < 2:
        return 0.0
    tokenized = [t.lower().split() for t in generateds]
    smoothie = SmoothingFunction().method4
    scores = []
    n = len(tokenized)
    for i in range(n):
        hyp = tokenized[i]
        refs = [tokenized[j] for j in range(n) if j != i]  
        score = sentence_bleu(
            refs,
            hyp,
            weights=tuple(1.0 / max_n for _ in range(max_n)),
            smoothing_function=smoothie
        )
        scores.append(score)     
    return sum(scores) / n


def compute_distinct_n(generated: str, n: int = 2) -> float:
    ngrams = get_ngrams(generated, n)
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def compute_repetition_n(generated: str, n: int = 2) -> float:
    ngrams = get_ngrams(generated, n)
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def compute_interestingness(
    task_description: str,
    generated_text: str,
    llm: BaseLanguageModel,
    language: str = "en"
) -> float:
    if language == "ru":
        chain = prompt_template_interestingness_en | llm
    else:
        chain = prompt_template_interestingness_en | llm

    try:
        response = chain.invoke({
            "task_description": task_description,
            "generated_text": generated_text
        })
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        answer = parse_llm_answer(answer, "answer")
        match = re.search(r"(\d*\.?\d+)", answer)
        if match:
            score = float(match.group(1))
            return score
        else:
            return 0.0
    except Exception as e:
        print(f"Error in interestingness evaluation: {e}")
        return 0.0



#-------------------------------------------
#---------metrics for RAG analysis----------
#-------------------------------------------


ragas_evaluation_llm = OllamaLLM(model="mistral:7b-instruct-q2_K", temperature=0.0, num_predict=1024, format="json")
ragas_evaluation_llm = LangchainLLMWrapper(ragas_evaluation_llm)

ragas_evaluation_embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
ragas_evaluation_embeddings = LangchainEmbeddingsWrapper(ragas_evaluation_embeddings)

ragas_run_config = RunConfig(max_workers=1, timeout=120)

def evaluate_ragas_metrics(
    question: str,
    answer: str,
    context: str,
    ground_truth: str
) -> Dict[str, float]:
    
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [split_context_by_lines(context)],
        "ground_truth": [ground_truth]   
    }
    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_similarity,
            answer_correctness
        ],
        llm=ragas_evaluation_llm,
        embeddings=ragas_evaluation_embeddings,
        run_config=ragas_run_config,
        show_progress=False
    )
    return { k: float(v[0] if isinstance(v, list) else v) for k, v in result.scores[0].items()}


def evaluate_bert_score_vs_source(generated_text: str, source_text: str, language: str = "en") -> float:
    P, R, F1 = bert_score_fn(
        cands=[generated_text],
        refs=[source_text],
        lang=language,
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return F1.mean().item()


def evaluate_bert_score_vs_reference(generated_text: str, reference_text: str, language: str = "en") -> float:
    P, R, F1 = bert_score_fn(
        cands=[generated_text],
        refs=[reference_text],
        lang=language,
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return F1.mean().item()


def compute_world_consistency(
    original_context: str,
    generated_text: str,
    llm: BaseLanguageModel,
    language: str = "en"
) -> float:
    if language == "ru":
        chain = prompt_template_world_consistency_en | llm
    else:
        chain = prompt_template_world_consistency_en | llm
    try:
        response = chain.invoke({
            "original_context": original_context,
            "generated_text": generated_text
        })
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        answer = parse_llm_answer(answer, "answer")
        match = re.search(r"(\d*\.?\d+)", answer)
        if match:
            score = float(match.group(1))
            return score
        else:
            return 0.0
    except Exception as e:
        print(f"Error in world consistency evaluation: {e}")
        return 0.0



#---------------------------------------------
#---------metrics for graph analysis----------
#---------------------------------------------

def get_edge_endpoints(edge: Edge) -> Optional[tuple[str, str]]:
    src = getattr(edge, 'source_id', None) or getattr(edge, 'source', None) or getattr(edge, 'from_node', None)
    tgt = getattr(edge, 'target_id', None) or getattr(edge, 'target', None) or getattr(edge, 'to_node', None)
    return (src, tgt) if src and tgt else None

def calculate_efficiency_metrics(graph: KnowledgeGraph) -> Dict[str, float]:
    nodes = graph.get_all_nodes()
    edges = graph.get_all_edges()
    node_ids = {n.id for n in nodes}
    num_nodes = len(nodes)
    num_edges = len(edges)
    if num_nodes == 0:
        return {"node count": 0.0, "edge count": 0.0, "average degree": 0.0, "average clustering coefficient": 0.0}

    degree = defaultdict(int)
    adj = defaultdict(set)

    for edge in edges:
        endpoints = get_edge_endpoints(edge)
        if endpoints:
            src, tgt = endpoints
            degree[src] += 1
            degree[tgt] += 1
            adj[src].add(tgt)
            adj[tgt].add(src)
    avg_degree = sum(degree.get(nid, 0) for nid in node_ids) / num_nodes

    clustering_coeffs = []
    for nid in node_ids:
        d = degree.get(nid, 0)
        if d < 2:
            clustering_coeffs.append(0.0)
            continue
        neighbors = list(adj[nid])
        triangles = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in adj[neighbors[i]]:
                    triangles += 1             
        c_v = (2 * triangles) / (d * (d - 1))
        clustering_coeffs.append(c_v)
    avg_clustering = sum(clustering_coeffs) / num_nodes

    return {
        "node count": float(num_nodes),
        "edge count": float(num_edges),
        "average degree": avg_degree,
        "average clustering coefficient": avg_clustering
    }


def calculate_suitability_metrics(graph: KnowledgeGraph) -> Dict[str, float]:
    nodes = graph.get_all_nodes()
    edges = graph.get_all_edges()

    type_counts = defaultdict(int)
    total_states = 0
    nodes_gt_2_states = 0

    for node in nodes:
        n_type = node.type.lower().strip() if node.type else ""
        type_counts[n_type] += 1

        total_states += len(node.states)
        if len(node.states) > 2:
            nodes_gt_2_states += 1

    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "characters": type_counts.get("character", 0),
        "groups": type_counts.get("group", 0),
        "locations": type_counts.get("location", 0),
        "location_elements": type_counts.get("location_element", 0),
        "events": type_counts.get("event", 0),
        "items": type_counts.get("item", 0),
        "total_states": total_states,
        "nodes_with_gt_2_states": nodes_gt_2_states
    }