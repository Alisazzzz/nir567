import json
from typing import List, Dict, Any, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import pandas as pd
import tqdm
import nir.tests.metrics as metrics

from nir.graph.graph_storages.networkx_graph import NetworkXGraph
from nir.graph.knowledge_graph import KnowledgeGraph
from nir.llm.manager import ModelManager
from nir.llm.providers import ModelConfig
from nir.tests.pipelines import BaseGraphPipeline, BasePipeline, GraphPipelineBasic, GraphPipelineFromNodes, StandardRAGPipeline
from nir.tests.metrics import compute_mauve, compute_distinct_n, compute_mauve_several, compute_repetition_n, compute_world_consistency, evaluate_bert_score_vs_reference, evaluate_bert_score_vs_source, evaluate_ragas_metrics
from nir.tests.test_datasets import TEST_DATA_TEXT1, TEST_TASKS, GRAPH_TEST_DATASET
from nir.data import loader

import os

def analyze_generation(
    generated_text: str,
    context: str,
    lore_summary: str,
    reference_text: str,
    query: str,
    category: str,
    llm: BaseLanguageModel,
    embedding_model: Embeddings,
    language: str = "en"
) -> Dict[str, Any]:

    metrics = {"category": category}
    # ragas_res = evaluate_ragas_metrics(query, generated_text, context, reference_text, llm, embedding_model)
    # metrics.update(ragas_res)

    metrics["bert_score_source"] = evaluate_bert_score_vs_source(generated_text, lore_summary, language)
    metrics["bert_score_reference"] = evaluate_bert_score_vs_reference(generated_text, reference_text, language)

    metrics["distinct_2"] = compute_distinct_n(generated_text, n=2)
    metrics["repetition_2"] = compute_repetition_n(generated_text, n=2)
    metrics["world_consistency"] = compute_world_consistency(lore_summary, generated_text, llm)
    return metrics


def run_generation_tests(
    dataset: Dict[str, Any],
    dataset_name: str,
    pipeline: BasePipeline,
    evaluation_llm: BaseLanguageModel,
    embedding_model: Embeddings,
    base_output_dir: str = "assets/outputs/test_results"
) -> pd.DataFrame:

    pipeline_dir = os.path.join(base_output_dir, pipeline.name)
    os.makedirs(pipeline_dir, exist_ok=True)
    output_json_path = os.path.join(pipeline_dir, f"{dataset_name}.json")

    language = dataset.get("language", "en")
    pipeline_data = {"language": language}

    if pipeline.name == "Basic LLM":
        pipeline_data["context"] = dataset["text_summary"]

    elif pipeline.name == "Standard RAG":
        path = dataset["path_to_text"]
        extension = os.path.splitext(path)[1].lower()
        if extension == ".csv":
            data = loader.loadCSV(path=path)
        elif extension == ".txt":
            data = loader.loadTXT(path=path)
        else:
            data = []
        chunks = loader.to_chunk(data)
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        pipeline_data["retriever"] = vectorstore.as_retriever(search_kwargs={"k": 3})

    else:
        graph = NetworkXGraph()
        graph.load(filepath=dataset["path_to_graph"])
        pipeline_data["graph"] = graph

        embedding_model = manager.create_embedding_model(
            name="embeddings", 
            option="hf_local", 
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        pipeline_data["embedding_model"] = embedding_model

    json_results: List[Dict[str, Any]] = []
    all_metrics: List[Dict[str, Any]] = []
    generateds = []
    references = []

    for task in tqdm.tqdm(dataset["tasks"], desc=f"Testing {pipeline.name} on {dataset_name}"):
        query = task["query"]
        reference = task["reference"]
        category = task.get("category", "default")
        pipeline_data["add_history"] = task.get("add_history", False)

        output = pipeline.generate(query, pipeline_data)
        gen_text = output.get("answer", "").strip()
        context = output.get("context", "").strip()

        if not gen_text:
            continue

        generateds.append(gen_text)
        references.append(reference)

        metrics = analyze_generation(
            generated_text=gen_text,
            context=context,
            lore_summary=dataset["text_summary"],
            reference_text=reference,
            query=query,
            category=category,
            llm=evaluation_llm,
            embedding_model=embedding_model,
            language=language
        )

        task_record = {
            "category": category,
            "query": query,
            "generation": gen_text,
            "metrics": metrics
        }
        json_results.append(task_record)
        all_metrics.append(metrics)

    mauve_result = compute_mauve_several(generateds, references)
    for metrics_dict in all_metrics:
        metrics_dict["mauve"] = mauve_result

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

    if not all_metrics:
        return pd.DataFrame()

    df = pd.DataFrame(all_metrics)
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if "category" in df.columns and len(df) > 0:
        cat_df = df.groupby("category")[num_cols].mean().reset_index()
    else:
        cat_df = df[num_cols].mean().to_frame().T
        cat_df["category"] = "default"

    overall = {col: df[col].mean() for col in num_cols}
    overall["category"] = "OVERALL"
    overall_df = pd.DataFrame([overall])

    final_df = pd.concat([cat_df, overall_df], ignore_index=True)
    cols_order = ["category"] + sorted([c for c in final_df.columns if c != "category"])
    final_df = final_df[cols_order]

    return final_df



#---------------------------------
#---------graph analysis----------
#---------------------------------

def analyze_graph(graph: KnowledgeGraph, expected_values: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    efficiency_metrics = metrics.calculate_efficiency_metrics(graph)
    suitability_metrics = metrics.calculate_suitability_metrics(graph)

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