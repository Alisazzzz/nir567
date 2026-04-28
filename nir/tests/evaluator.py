#All stuff for run several metrics on graph or generation is here



#--------------------------
#---------imports----------
#--------------------------

import os
from typing import List, Dict, Any, Tuple
import pandas as pd
import tqdm

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
import pandas as pd
import tqdm

from nir.graph.knowledge_graph import KnowledgeGraph
from nir.llm.manager import ModelManager
from nir.tests.pipelines import BaseGraphPipeline
from nir.tests.metrics import (
    compute_interestingness, 
    compute_distinct_n, 
    compute_repetition_n, 
    compute_world_consistency, 
    evaluate_bert_score_vs_reference, 
    evaluate_bert_score_vs_source, 
    evaluate_ragas_metrics,

    calculate_efficiency_metrics,
    calculate_suitability_metrics   
)
from nir.data import loader



#--------------------------------
#---------text analysis----------
#--------------------------------

def analyze_generation(
    generated_text: str,
    context: str,
    lore_summary: str,
    reference_text: str,
    query: str,
    category: str,
    evaluation_llm: BaseLanguageModel,
    language: str = "en"
) -> Dict[str, Any]:

    metrics = {"category": category}
    ragas_res = evaluate_ragas_metrics(query, generated_text, context, reference_text)
    metrics.update(ragas_res)
    metrics["bert_score_source"] = evaluate_bert_score_vs_source(generated_text, lore_summary, language)
    metrics["bert_score_reference"] = evaluate_bert_score_vs_reference(generated_text, reference_text, language)
    metrics["world_consistency"] = compute_world_consistency(lore_summary, generated_text, evaluation_llm)

    metrics["distinct_2"] = compute_distinct_n(generated_text, n=2)
    metrics["repetition_2"] = compute_repetition_n(generated_text, n=2)
    metrics["interestingness"] = compute_interestingness(query, generated_text, evaluation_llm)

    return metrics



#---------------------------------
#---------graph analysis----------
#---------------------------------

def analyze_graph(graph: KnowledgeGraph, expected_values: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    efficiency_metrics = calculate_efficiency_metrics(graph)
    suitability_metrics = calculate_suitability_metrics(graph)

    efficiency_df = pd.DataFrame([ {"Metric": k, "Value": v} for k, v in efficiency_metrics.items() ])

    all_metric_names = sorted(set(suitability_metrics.keys()) | set(expected_values.keys()))
    suitability_rows = []
    for metric in all_metric_names:
        actual = suitability_metrics.get(metric, 0.0)
        expected = expected_values.get(metric, 0.0)
        squared_error = abs(((actual - expected) / (expected + 1)) * 100)
        suitability_rows.append({
            "Metric": metric,
            "Model Result": actual,
            "Expected Result": expected,
            "Squared Error": squared_error
        })
    suitability_df = pd.DataFrame(suitability_rows)

    return efficiency_df, suitability_df


def run_graph_tests(dataset: List[Dict[str, Any]], pipeline: BaseGraphPipeline) -> Tuple[pd.DataFrame, pd.DataFrame]:

    eff_dfs = []
    suit_dfs = []

    for item in tqdm.tqdm(dataset):
        path = item["path"]
        language = item["language"]

        embedding_model_info = item["embedding_model_options"]
        embedding_model = manager.create_embedding_model(
            name=embedding_model_info["name"], 
            option=embedding_model_info["option"], 
            model_name=embedding_model_info["model_name"])
        
        expected_values = item.get("expected_values", {})

        extension = os.path.splitext(path)[1].lower()
        if extension == ".csv":
            data = loader.loadCSV(path=path)
        elif extension == ".txt":
            data = loader.loadTXT(path=path)

        chunks = loader.to_chunk_unique_id(docs=data, start_chunk_id=0)
        graph = pipeline.extract_graph(chunks=chunks, embedding_model=embedding_model, language=language)

        eff_df, suit_df = analyze_graph(graph, expected_values)
        print(eff_df)
        print(suit_df)
        eff_dfs.append(eff_df)
        suit_dfs.append(suit_df)

    if eff_dfs:
        avg_eff_df = pd.concat(eff_dfs).groupby("Metric")["Value"].mean().reset_index()
        avg_eff_df.columns = ["Metric", "Average Value"]
    else:
        avg_eff_df = pd.DataFrame(columns=["Metric", "Average Value"])

    if suit_dfs:
        all_suit = pd.concat(suit_dfs)
        avg_suit_df = all_suit.groupby("Metric").agg({
            "Model Result": "mean",
            "Expected Result": "mean",
            "Squared Error": "mean"
        }).reset_index()
    else:
        avg_suit_df = pd.DataFrame(columns=["Metric", "Model Result", "Expected Result", "Squared Error"])
    return avg_eff_df, avg_suit_df



manager = ModelManager()

# model_config = ModelConfig(model_name="hf.co/VlSav/Vikhr-Nemo-12B-Instruct-R-21-09-24-Q4_K_M-GGUF:latest", temperature=0)
# instruct_model = manager.create_chat_model(name="graph_extraction", option="ollama", config=model_config)

# print("GRAPH RESULTS BASIC")
# basic_pipeline = GraphPipelineBasic(
#     llm=instruct_model,
#     graph_class=NetworkXGraph,
#     preserve_all_data=True,
#     name="Basic pipeline"
# )
# results_1, results_2 = run_graph_tests(dataset=GRAPH_TEST_DATASET, pipeline=basic_pipeline)
# print("Efficiency")
# print(results_1)
# print("Suitability")
# print(results_2)

# print("GRAPH RESULTS FROM NODES")
# pipeline_from_nodes = GraphPipelineFromNodes(
#     llm=instruct_model,
#     graph_class=NetworkXGraph,
#     preserve_all_data=True,
#     name="Pipeline from nodes"
# )
# results_3, results_4 = run_graph_tests(dataset=GRAPH_TEST_DATASET, pipeline=pipeline_from_nodes)
# print("Efficiency")
# print(results_3)
# print("Suitability")
# print(results_4)

manager = ModelManager()
# model_config = ModelConfig(
#     model_name="hf.co/VlSav/Vikhr-Nemo-12B-Instruct-R-21-09-24-Q4_K_M-GGUF:latest", 
#     temperature=0.7
# )
# llm = manager.create_chat_model(name="test_model", option="ollama", config=model_config)

# embed_model = manager.create_embedding_model(
#     name="embeddings", 
#     option="hf_local", 
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# run_generation_tests(
#     dataset=TEST_DATA_TEXT1,
#     dataset_name="lore description",
#     pipeline=StandardRAGPipeline(llm=llm),
#     evaluation_llm=llm,
#     embedding_model=embed_model
# )