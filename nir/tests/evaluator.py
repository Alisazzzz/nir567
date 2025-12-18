import json
import csv
from typing import List, Dict, Any

from langchain_core.language_models import BaseLanguageModel

from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.llm.manager import ModelManager
from nir.llm.providers import ModelConfig
from nir.tests.pipelines import BasePipeline, BasicLLMPipeline, StandardRAGPipeline, ThisNIRPipeline
from nir.tests.metrics import compute_mauve, compute_distinct_n, compute_repetition_n, compute_world_consistency
from nir.tests.datasets import TEST_TASKS

import os
os.environ["HF_HUB_TIMEOUT"] = "120"

def compute_metric(metric_type: str, task: Dict[str, Any], response: str, llm: BaseLanguageModel) -> float:
    match metric_type:
        case "mauve":
            return compute_mauve(response, task["expected"])
        case "distinct-n":
            return compute_distinct_n(response, n=2)
        case "repetition-n":
            return compute_repetition_n(response, n=2)
        case "world_consistency":
            return compute_world_consistency(original_context=task["world_context"], generated_text=response, llm=llm)
        case _:
            print(f"Unknown metric: {metric_type}")

def save_results(output_path: str, results: List[Dict[str, str]]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"Results saved to {output_path}")

    json_path = output_path.replace(".csv", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results also saved to {json_path}")

def print_summary(results: List[Dict[str, str]]) -> None:
    summary = {}
    for r in results:
        key = (r["pipeline"], r["metric"])
        if key not in summary:
            summary[key] = []
        summary[key].append(r["score"])

    print("\n=== SUMMARY ===")
    for (pipeline, metric), scores in summary.items():
        avg = sum(scores) / len(scores)
        print(f"{pipeline} | {metric}: {avg:.4f}")

def run(pipelines: List[BasePipeline], tasks: List[Dict[str, Any]], llm: BaseLanguageModel) -> List[Dict[str, str]]:
    print("Starting evaluation...\n")
    results = []

    for task in tasks:
        print(f"Processing task: {task["id"]}")
        for pipeline in pipelines:
            print(f"Running {pipeline.name}")
            try:
                response = pipeline.generate(task["query"])
            except Exception as e:
                print(f"Error in {pipeline.name}: {e}")
                response = ""

            for metric in task["metric"]:
                score = compute_metric(metric, task, response, llm)
                result_entry = {
                    "task_id": task["id"],
                    "pipeline": pipeline.name,
                    "query": task["query"],
                    "response": response,
                    "metric": metric,
                    "score": score
                }
                results.append(result_entry)
                print(f"Score ({metric}): {score:.4f}")
        print()
    
    return results

manager = ModelManager()

embedding_model = manager.create_embedding_model(name="embeddings", option="ollama", model_name="nomic-embed-text:v1.5")
model_config = ModelConfig(model_name="llama3.2:latest", temperature=0.8)
chat_model = manager.create_chat_model(name="basic_chat", option="ollama", config=model_config)
llm_judge = chat_model 

basic_context = """
    Morgiana is a female slave living in the household of Ali Baba in a Persian town. 
    She is known for her intelligence, attentiveness, and loyalty to her master. 
    Morgiana takes part in daily household duties, moves freely around the house, runs errands, and interacts with local craftsmen and merchants, 
    such as the cobbler Mustapha. She is observant and cautious, often staying awake late at night and noticing small details others might miss. 
    Although her life is outwardly ordinary and centered on domestic responsibilities, Morgiana is capable of decisive action when danger arises. 
    She ultimately becomes a trusted and central figure in Ali Baba’s household and is later rewarded for her faithfulness.
"""

#pipelines
basic = BasicLLMPipeline(
    llm=chat_model,
    context=basic_context,
    name="Basic LLM"
)
rag = StandardRAGPipeline(
    llm=chat_model,
    embedding_model=embedding_model,
    filepath="assets/documents/ali baba, or the forty thieves.txt",
    name="Standard RAG"
)

graph = NetworkXGraph()
graph.load(filepath="assets/graphs/graph_ali_baba.json")

nir = ThisNIRPipeline(
    llm=chat_model,
    embedding_model=embedding_model,
    knowledge_graph=graph,
    use_llm_for_timestamps=False,
    add_history=True,
    name="This NIR RAG"
)

pipelines = [basic, rag, nir]
tasks = TEST_TASKS
result = run(pipelines=pipelines, tasks=tasks, llm=llm_judge)
save_results("assets/outputs/evaluation.csv", result)
print_summary(result)