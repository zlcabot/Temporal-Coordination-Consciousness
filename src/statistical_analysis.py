"""
Statistical Analysis Module

Functions for computing effect sizes, confidence intervals, and
performing statistical tests on coordination data.

References:
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    Cabot, Z. (2025). Hemispheric Dialogue Across Species.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, List, Optional
import pandas as pd


def compute_cohens_d(
    group1: np.ndarray, 
    group2: np.ndarray,
    paired: bool = True
) -> float:
    """
    Compute Cohen's d effect size.
    
    For paired data: d = mean_diff / std_diff
    For independent data: d = (mean1 - mean2) / pooled_std
    
    Parameters
    ----------
    group1 : np.ndarray
        First group values
    group2 : np.ndarray
        Second group values
    paired : bool
        Whether data is paired (default: True)
    
    Returns
    -------
    float
        Cohen's d effect size
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    if paired:
        diff = group1 - group2
        d = np.mean(diff) / np.std(diff, ddof=1)
    else:
        n1, n2 = len(group1), len(group2)
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d


def compute_effect_size_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    paired: bool = True,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float, float]:
    """
    Compute Cohen's d with bootstrap confidence interval.
    
    Parameters
    ----------
    group1 : np.ndarray
        First group values
    group2 : np.ndarray
        Second group values
    paired : bool
        Whether data is paired
    confidence : float
        Confidence level (default: 0.95)
    n_bootstrap : int
        Number of bootstrap samples (default: 1000)
    
    Returns
    -------
    tuple
        (d, ci_lower, ci_upper)
    """
    d = compute_cohens_d(group1, group2, paired)
    
    # Bootstrap
    n = len(group1)
    d_bootstrap = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        d_boot = compute_cohens_d(group1[idx], group2[idx], paired)
        d_bootstrap.append(d_boot)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(d_bootstrap, 100 * alpha / 2)
    ci_upper = np.percentile(d_bootstrap, 100 * (1 - alpha / 2))
    
    return d, ci_lower, ci_upper


def compute_eta_squared(
    groups: List[np.ndarray]
) -> float:
    """
    Compute eta-squared (η²) effect size for ANOVA.
    
    η² = SS_between / SS_total
    
    Parameters
    ----------
    groups : list of np.ndarray
        List of group data arrays
    
    Returns
    -------
    float
        Eta-squared effect size
    """
    # Combine all data
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    # SS_total
    ss_total = np.sum((all_data - grand_mean) ** 2)
    
    # SS_between
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    
    return eta_sq


def paired_ttest_with_effect_size(
    x: np.ndarray,
    y: np.ndarray
) -> Dict:
    """
    Perform paired t-test with effect size and confidence interval.
    
    Parameters
    ----------
    x : np.ndarray
        First condition values
    y : np.ndarray
        Second condition values
    
    Returns
    -------
    dict
        Statistical results including t, df, p, d, and CI
    """
    t_stat, p_value = stats.ttest_rel(x, y)
    d, ci_lower, ci_upper = compute_effect_size_ci(x, y, paired=True)
    
    return {
        't_statistic': t_stat,
        'degrees_of_freedom': len(x) - 1,
        'p_value': p_value,
        'cohens_d': d,
        'cohens_d_ci': (ci_lower, ci_upper),
        'mean_difference': np.mean(x - y),
        'std_difference': np.std(x - y, ddof=1)
    }


def wilcoxon_test(
    x: np.ndarray,
    y: np.ndarray
) -> Dict:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    
    Parameters
    ----------
    x : np.ndarray
        First condition values
    y : np.ndarray
        Second condition values
    
    Returns
    -------
    dict
        Statistical results including W and p
    """
    stat, p_value = stats.wilcoxon(x, y)
    
    return {
        'W_statistic': stat,
        'p_value': p_value,
        'n': len(x)
    }


def subject_level_aggregation(
    df: pd.DataFrame,
    subject_col: str = 'subject',
    value_cols: List[str] = None
) -> pd.DataFrame:
    """
    Aggregate epoch-level data to subject level.
    
    Parameters
    ----------
    df : pd.DataFrame
        Epoch-level data
    subject_col : str
        Column name for subject ID
    value_cols : list
        Columns to aggregate (default: all numeric)
    
    Returns
    -------
    pd.DataFrame
        Subject-level summary
    """
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if subject_col in value_cols:
            value_cols.remove(subject_col)
    
    agg_funcs = {col: ['mean', 'std', 'count'] for col in value_cols}
    
    subject_summary = df.groupby(subject_col).agg(agg_funcs)
    subject_summary.columns = ['_'.join(col).strip() for col in subject_summary.columns]
    
    return subject_summary


def compute_consistency(
    values: np.ndarray,
    threshold: float = 1.0,
    direction: str = 'greater'
) -> Tuple[int, int, float]:
    """
    Compute consistency of effect direction.
    
    Parameters
    ----------
    values : np.ndarray
        Values to assess
    threshold : float
        Threshold value
    direction : str
        'greater' or 'less'
    
    Returns
    -------
    tuple
        (n_consistent, n_total, percentage)
    """
    if direction == 'greater':
        consistent = np.sum(values > threshold)
    else:
        consistent = np.sum(values < threshold)
    
    total = len(values)
    pct = 100 * consistent / total if total > 0 else 0
    
    return consistent, total, pct


def full_statistical_report(
    inter_values: np.ndarray,
    intra_values: np.ndarray
) -> Dict:
    """
    Generate complete statistical report for inter vs intra comparison.
    
    Parameters
    ----------
    inter_values : np.ndarray
        Inter-hemispheric Φc values
    intra_values : np.ndarray
        Intra-hemispheric Φc values
    
    Returns
    -------
    dict
        Complete statistical report
    """
    DI_values = inter_values / intra_values
    
    # Descriptives
    report = {
        'descriptives': {
            'inter_mean': np.mean(inter_values),
            'inter_std': np.std(inter_values, ddof=1),
            'intra_mean': np.mean(intra_values),
            'intra_std': np.std(intra_values, ddof=1),
            'DI_mean': np.mean(DI_values),
            'DI_std': np.std(DI_values, ddof=1),
            'DI_median': np.median(DI_values),
            'n': len(inter_values)
        }
    }
    
    # Parametric test
    report['parametric'] = paired_ttest_with_effect_size(inter_values, intra_values)
    
    # Non-parametric test
    report['nonparametric'] = wilcoxon_test(inter_values, intra_values)
    
    # Consistency
    n_consistent, n_total, pct = compute_consistency(DI_values, threshold=1.0, direction='greater')
    report['consistency'] = {
        'DI_gt_1_count': n_consistent,
        'DI_gt_1_total': n_total,
        'DI_gt_1_percentage': pct
    }
    
    return report


# Example usage
if __name__ == '__main__':
    # Generate sample data matching paper statistics
    np.random.seed(42)
    n = 1000
    
    inter = np.random.normal(0.606, 0.203, n)
    intra = np.random.normal(0.222, 0.160, n)
    
    inter = np.clip(inter, 0.05, 0.99)
    intra = np.clip(intra, 0.02, 0.95)
    
    report = full_statistical_report(inter, intra)
    
    print("=== Statistical Report ===")
    print(f"\nDescriptives:")
    for k, v in report['descriptives'].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print(f"\nParametric (paired t-test):")
    print(f"  t = {report['parametric']['t_statistic']:.2f}")
    print(f"  p = {report['parametric']['p_value']:.2e}")
    print(f"  Cohen's d = {report['parametric']['cohens_d']:.2f}")
    
    print(f"\nConsistency:")
    print(f"  DI > 1: {report['consistency']['DI_gt_1_percentage']:.1f}%")
