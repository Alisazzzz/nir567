#All stuff for run several metrics on graph or generation is here



#--------------------------
#---------imports----------
#--------------------------

from typing import Dict, Any, Literal, Tuple, Optional, Union
import numpy as np
import pandas as pd

from langchain_core.language_models import BaseLanguageModel
from scipy import stats

from nir.graph.knowledge_graph import KnowledgeGraph
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



#-----------------------------------
#---------results analysis----------
#-----------------------------------

def extend_dataset(
    df: pd.DataFrame,
    metric_column: str,
    n_replications: int = 100,
    noise_std_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> pd.DataFrame:

    if seed is not None:
        np.random.seed(seed)
    original_values = df[metric_column].dropna()
    if len(original_values) == 0:
        return df.drop(columns=[metric_column], errors='ignore')
    observed_std = original_values.std()
    perturbation_std = observed_std * noise_std_ratio

    result_parts = []
    df_original = df.copy()
    result_parts.append(df_original)

    for rep_id in range(n_replications):
        df_perturbed = df.copy()
        noise = np.random.normal(0, perturbation_std, size=len(df))
        df_perturbed[metric_column] = df_perturbed[metric_column] + noise
        result_parts.append(df_perturbed)
    
    result_df = pd.concat(result_parts, ignore_index=True)
    return result_df

def simulate_paired_dataset(
    df_baseline: pd.DataFrame,
    df_proposed: pd.DataFrame,
    metric_column: str,
    n_simulations: int = 1000,
    seed: Optional[int] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if seed is not None:
        np.random.seed(seed)

    common_idx = df_baseline.index.intersection(df_proposed.index)
    baseline = df_baseline.loc[common_idx, metric_column].reset_index(drop=True)
    proposed = df_proposed.loc[common_idx, metric_column].reset_index(drop=True)
    mask = baseline.notna() & proposed.notna()
    baseline = baseline[mask].to_numpy()
    proposed = proposed[mask].to_numpy()
    if len(baseline) < 2:
        raise ValueError("Not enough paired observations")
    differences = proposed - baseline
    mu_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    mu_base = np.mean(baseline)
    std_base = np.std(baseline, ddof=1)
    simulated_baseline = np.random.normal(
        loc=mu_base,
        scale=std_base,
        size=n_simulations
    )
    simulated_diff = np.random.normal(
        loc=mu_diff,
        scale=std_diff,
        size=n_simulations
    )
    simulated_proposed = simulated_baseline + simulated_diff
    df_base_sim = pd.DataFrame({
        metric_column: simulated_baseline
    })
    df_prop_sim = pd.DataFrame({
        metric_column: simulated_proposed
    })
    return df_base_sim, df_prop_sim

def compare_pipelines_directional(
    df_baseline: pd.DataFrame,
    df_proposed: pd.DataFrame,
    metric_column: str,
    alternative: Literal['greater', 'less'] = 'greater',
    alpha: float = 0.05,
    normality_test_alpha: float = 0.05,
    paired_by_index: bool = True
) -> Dict[str, Union[str, float, bool]]:

    if paired_by_index:
        common_idx = df_baseline.index.intersection(df_proposed.index)
        baseline_vals = df_baseline.loc[common_idx, metric_column].dropna()
        proposed_vals = df_proposed.loc[common_idx, metric_column].dropna()
    else:
        min_len = min(len(df_baseline), len(df_proposed))
        baseline_vals = df_baseline[metric_column].dropna().iloc[:min_len]
        proposed_vals = df_proposed[metric_column].dropna().iloc[:min_len]

    baseline_vals = baseline_vals.reset_index(drop=True)
    proposed_vals = proposed_vals.reset_index(drop=True)
    
    mask = baseline_vals.notna() & proposed_vals.notna()
    baseline_vals = baseline_vals[mask]
    proposed_vals = proposed_vals[mask]
    
    if len(baseline_vals) < 2:
        return {}

    differences = proposed_vals - baseline_vals
    _, normality_p = stats.shapiro(differences)
    is_normal = normality_p > normality_test_alpha

    if is_normal:
        # Paired t-test (one-tailed)
        t_stat, p_two_tailed = stats.ttest_rel(proposed_vals, baseline_vals)
        if alternative == 'greater':
            p_value = p_two_tailed / 2 if t_stat > 0 else 1 - p_two_tailed / 2
        elif alternative == 'less':
            p_value = p_two_tailed / 2 if t_stat < 0 else 1 - p_two_tailed / 2
        else:
            raise ValueError("alternative must be 'greater' or 'less'")
        test_used = 'Paired t-test (one-tailed)'
        statistic = t_stat
    else:
        # Wilcoxon signed-rank test (one-tailed)
        try:
            stat, p_value = stats.wilcoxon(proposed_vals, baseline_vals, alternative=alternative)
            test_used = 'Wilcoxon signed-rank (one-tailed)'
            statistic = stat
        except TypeError:
            stat, p_two_tailed = stats.wilcoxon(proposed_vals, baseline_vals, mode='exact')
            mean_diff = np.mean(differences)
            if alternative == 'greater':
                p_value = p_two_tailed / 2 if mean_diff > 0 else 1 - p_two_tailed / 2
            elif alternative == 'less':
                p_value = p_two_tailed / 2 if mean_diff < 0 else 1 - p_two_tailed / 2
            test_used = 'Wilcoxon signed-rank (one-tailed, manual one-tail)'
            statistic = stat

    mean_diff = np.mean(differences)
    significant = p_value < alpha
    
    return {
        'test_used': test_used,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': significant,
        'normality_p': float(normality_p),
        'is_normal': is_normal,
        'mean_diff': float(mean_diff),
        'n_pairs': len(differences)
    }
def compute_effect_size(
    x: Union[pd.Series, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    effect_type: Literal['cohens_d', 'rank_biserial'] = 'cohens_d',
    paired: bool = True
) -> Dict[str, Union[float, str]]:
    x_clean = np.asarray(pd.Series(x).dropna())
    y_clean = np.asarray(pd.Series(y).dropna())
    
    if len(x_clean) < 2 or len(y_clean) < 2:
        raise ValueError("Not enough data to evaluate effect size")
    if paired:
        min_len = min(len(x_clean), len(y_clean))
        x_clean = x_clean[:min_len]
        y_clean = y_clean[:min_len]
        mask = ~np.isnan(x_clean) & ~np.isnan(y_clean)
        x_clean = x_clean[mask]
        y_clean = y_clean[mask]
        
    if len(x_clean) < 2:
        raise ValueError("Not enough paired data to evaluate effect size")
    
    if effect_type == 'cohens_d':
        differences = x_clean - y_clean
        std_diff = np.std(differences, ddof=1)
        if std_diff == 0:
            return {'effect_size': 0.0, 'type': "Cohen's d (paired)"}
        d = np.mean(differences) / std_diff
        return {
            'effect_size': float(d),
            'type': "Cohen's d (paired)"
        }
    
    elif effect_type == 'rank_biserial':
        from scipy.stats import rankdata
        
        differences = x_clean - y_clean
        abs_diffs = np.abs(differences)
        non_zero_mask = abs_diffs > 1e-10
        if not np.any(non_zero_mask):
            return {'effect_size': 0.0, 'type': 'rank-biserial'}
        
        ranks = rankdata(abs_diffs[non_zero_mask])
        signed_ranks = ranks * np.sign(differences[non_zero_mask])
        
        W_plus = np.sum(signed_ranks[signed_ranks > 0])
        W_minus = np.abs(np.sum(signed_ranks[signed_ranks < 0]))
        n = len(signed_ranks)
        r_rb = (W_plus - W_minus) / (n * (n + 1) / 2)
        
        return {
            'effect_size': float(r_rb),
            'type': 'rank-biserial correlation'
        }
    
    else:
        raise ValueError("effect_type must be 'cohens_d' or 'rank_biserial'")
    
