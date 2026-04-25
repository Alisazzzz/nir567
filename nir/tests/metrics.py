#All stuff for certain metrics is here


#--------------------------
#---------imports----------
#--------------------------


import re
from typing import Dict, List, Optional
from collections import Counter, defaultdict

from langchain.schema import Generation, LLMResult
from langchain_core.embeddings import Embeddings
import torch
from datasets import Dataset
from bert_score import score as bert_score_fn

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import mauve
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

from nir.graph.graph_structures import Edge
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.tests import testing_prompts

import re
import json
from typing import Any
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from ragas import RunConfig, evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,  # актуальное имя вместо AnswerAccuracy
)


import re
import json
from typing import Any, Optional, List, Union
from langchain_ollama import OllamaLLM
from langchain_core.prompt_values import StringPromptValue, ChatPromptValue
from langchain_core.messages import BaseMessage
from ragas.run_config import RunConfig

import re
from typing import Any, Optional, List
from langchain_ollama import OllamaLLM
from langchain_core.outputs import LLMResult, Generation
from ragas.run_config import RunConfig

class StrictJSONOllamaWrapper:
    """
    Обертка для Ollama, которая заставляет модель возвращать строгий JSON.
    Правильно обрабатывает callbacks и другие параметры.
    """
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        self._run_config = None
        
    def set_run_config(self, run_config: RunConfig) -> None:
        """Устанавливает конфигурацию запуска"""
        self._run_config = run_config
        if hasattr(self.llm, 'set_run_config'):
            self.llm.set_run_config(run_config)
    
    def _clean_json(self, text: str) -> str:
        """Очищает JSON от markdown и лишнего текста"""
        if not text:
            return "{}"
        
        text = str(text)
        
        # Убираем markdown-блоки
        text = re.sub(r'```json\s*\n?', '', text)
        text = re.sub(r'```\s*\n?', '', text)
        
        # Ищем первую { или [
        start = text.find('{')
        if start == -1:
            start = text.find('[')
        if start == -1:
            return "{}"
            
        # Ищем закрывающую скобку
        stack = []
        for i in range(start, len(text)):
            char = text[i]
            if char in '{[':
                stack.append(char)
            elif char == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
                    if not stack:
                        return text[start:i+1]
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                    if not stack:
                        return text[start:i+1]
        return text[start:]
    
    def _add_json_instruction(self, prompt: str) -> str:
        """Добавляет инструкцию о формате JSON"""
        instruction = "\n\nCRITICAL: Respond with ONLY valid JSON. No explanations, no markdown formatting. Start directly with { or [."
        return prompt + instruction
    
    def generate(self, prompts: List[str], **kwargs) -> LLMResult:
        """
        Генерирует ответы с очисткой JSON.
        Важно: НЕ передаем callbacks отдельно, они уже в kwargs.
        """
        # Убираем 'callbacks' из kwargs если он есть в виде отдельного параметра
        # (хотя в **kwargs он уже должен быть)
        modified_prompts = [self._add_json_instruction(p) for p in prompts]
        
        # Вызываем оригинальный generate
        # Передаем все kwargs как есть, без извлечения callbacks
        result = self.llm.generate(modified_prompts, **kwargs)
        
        # Очищаем JSON в каждом ответе
        for gen_list in result.generations:
            for gen in gen_list:
                if hasattr(gen, 'text'):
                    gen.text = self._clean_json(gen.text)
        
        return result
    
    async def agenerate(self, prompts: List[str], **kwargs) -> LLMResult:
        """
        Асинхронная версия generate.
        """
        modified_prompts = [self._add_json_instruction(p) for p in prompts]
        
        if hasattr(self.llm, 'agenerate'):
            result = await self.llm.agenerate(modified_prompts, **kwargs)
        else:
            # Fallback на синхронный
            import asyncio
            import concurrent.futures
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(
                    pool,
                    lambda: self.llm.generate(modified_prompts, **kwargs)
                )
        
        # Очищаем JSON
        for gen_list in result.generations:
            for gen in gen_list:
                if hasattr(gen, 'text'):
                    gen.text = self._clean_json(gen.text)
        
        return result
    
    # Проксируем остальные атрибуты
    def __getattr__(self, name):
        return getattr(self.llm, name)






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

#----------------------------------------------------
#---------metrics for text quality analysis----------
#----------------------------------------------------

def compute_mauve(generated: str, reference: str, model_id: str = "gpt2") -> float:
    if not generated.strip() or not reference.strip():
        return 0.0

    out = mauve.compute_mauve(
        p_text=reference,
        q_text=generated,
        device_id=0,
        verbose=False,
    )
    return float(out.mauve)

def compute_mauve_several(generated: list[str], reference: list[str], model_id: str = "gpt2") -> float:
    out = mauve.compute_mauve(
        p_text=reference,
        q_text=generated,
        device_id=0,
        verbose=False,
    )
    return float(out.mauve)

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


def compute_world_consistency(
    original_context: str,
    generated_text: str,
    llm: BaseLanguageModel
) -> float:
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", testing_prompts.SYSTEM_PROMPT_WORLD_CONSISTENCY_EN),
        ("human", (
            "Original game world description:\n{original_context}\n\n"
            "Newly created text:\n{generated_text}\n\n" ))
    ])

    chain = prompt_template | llm
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
            return min(max(score, 0.0), 1.0)
        else:
            return 0.0
    except Exception as e:
        print(f"Error in world consistency evaluation: {e}")
        return 0.0



#-------------------------------------------
#---------metrics for RAG analysis----------
#-------------------------------------------

def evaluate_ragas_metrics(
    question: str,
    answer: str,
    context: str,
    ground_truth: str,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
) -> Dict[str, float]:
    
    data = [
        {
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "contexts": split_context_by_lines(context)
        }
    ]

    dataset = Dataset.from_list(data)
    print(dataset)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    base_llm = OllamaLLM(
        model="hf.co/VlSav/Vikhr-Nemo-12B-Instruct-R-21-09-24-Q4_K_M-GGUF:latest",  # например, "llama3.2", "mistral", "phi3"
        temperature=0,  # Важно: 0 для детерминированных ответов
        num_ctx=4096,   # Увеличиваем контекст
    )

    # Embeddings
    base_embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")  # или другая модель

    # Оборачиваем для Ragas
    llm = LangchainLLMWrapper(base_llm)
    embeddings = LangchainEmbeddingsWrapper(base_embeddings)

    # Настройка RunConfig - критически важно для Ollama!
    run_config = RunConfig(
        max_workers=2,      # Меньше параллельных запросов
        timeout=80,        # Больше таймаут для маленьких моделей
    )

    # Теперь evaluate с конфигом
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )
        
    print(result.scores[0])
    
    return { k: float(v[0] if isinstance(v, list) else v) for k, v in result.scores[0].items()}


def evaluate_bert_score_vs_source(generated_text: str, source_text: str, language: str = "en") -> float:
    P, R, F1 = bert_score_fn(
        cands=[generated_text],
        refs=[source_text],
        lang=language,
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
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