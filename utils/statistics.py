"""
Statistical Analysis Module for Student Performance Analysis System.
Provides descriptive statistics, correlation calculations, and interpretation functions.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def compute_descriptive_stats(series: pd.Series) -> Dict[str, float]:
    """
    Computes comprehensive descriptive statistics for a numerical series using
    Pandas, NumPy, and SciPy.
    """
    s = series.dropna()
    if len(s) == 0:
        return {}

    mode_res = stats.mode(s, keepdims=True)
    mode_val = float(mode_res.mode[0]) if len(mode_res.mode) > 0 else float(s.median())
    
    return {
        "Count": int(len(s)),
        "Mean": round(float(np.mean(s)), 2),
        "Median": round(float(np.median(s)), 2),
        "Mode": round(mode_val, 2),
        "Variance": round(float(np.var(s, ddof=1)), 2),
        "Std Deviation": round(float(np.std(s, ddof=1)), 2),
        "Minimum": round(float(np.min(s)), 2),
        "25th Percentile (Q1)": round(float(np.percentile(s, 25)), 2),
        "75th Percentile (Q3)": round(float(np.percentile(s, 75)), 2),
        "Maximum": round(float(np.max(s)), 2),
        "Skewness": round(float(stats.skew(s)), 2),
        "Kurtosis": round(float(stats.kurtosis(s)), 2)
    }


def generate_numerical_summary_table(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Builds an analytical summary table for the specified numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    records = []
    for col in columns:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                mode_res = stats.mode(s, keepdims=True)
                mode_val = float(mode_res.mode[0]) if len(mode_res.mode) > 0 else float(s.median())
                records.append({
                    "Feature": col,
                    "Count": int(len(s)),
                    "Mean": round(float(s.mean()), 2),
                    "Median": round(float(s.median()), 2),
                    "Std Dev": round(float(s.std()), 2),
                    "Variance": round(float(s.var()), 2),
                    "Min": round(float(s.min()), 2),
                    "Q1 (25%)": round(float(s.quantile(0.25)), 2),
                    "Q3 (75%)": round(float(s.quantile(0.75)), 2),
                    "Max": round(float(s.max()), 2),
                    "Skewness": round(float(stats.skew(s)), 2)
                })
    return pd.DataFrame(records)


def compute_correlation_with_interpretation(
    x: pd.Series,
    y: pd.Series,
    x_name: str = "Variable X",
    y_name: str = "Variable Y"
) -> Dict[str, any]:
    """
    Computes Pearson correlation coefficient (r) and p-value using SciPy.
    Generates an honest, academically sound non-causal interpretation.
    """
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return {
            "r": 0.0,
            "p_value": 1.0,
            "strength": "Insufficient Data",
            "direction": "None",
            "interpretation": "Insufficient data to calculate a meaningful correlation."
        }
        
    r_val, p_val = stats.pearsonr(valid["x"], valid["y"])
    r_val = float(r_val)
    p_val = float(p_val)
    
    # Assess strength
    abs_r = abs(r_val)
    if abs_r >= 0.70:
        strength = "Strong"
    elif abs_r >= 0.40:
        strength = "Moderate"
    elif abs_r >= 0.20:
        strength = "Weak"
    else:
        strength = "Very Weak / Negligible"
        
    direction = "Positive" if r_val > 0 else "Negative" if r_val < 0 else "Neutral"
    
    # Statistical significance note
    sig_note = "statistically significant (p < 0.05)" if p_val < 0.05 else "not statistically significant (p >= 0.05)"
    
    # Academic interpretation
    if abs_r < 0.20:
        interp = (
            f"There is little to no linear relationship between {x_name} and {y_name} (r = {r_val:.2f}, p = {p_val:.4f}). "
            f"Variations in {x_name} do not linearly correspond to changes in {y_name}."
        )
    else:
        trend_desc = "higher values of both tend to co-occur" if r_val > 0 else f"higher {x_name} tends to accompany lower {y_name}"
        interp = (
            f"There is a {strength.lower()} {direction.lower()} relationship between {x_name} and {y_name} "
            f"(Pearson r = {r_val:.2f}, {sig_note}). This suggests that {trend_desc}. "
            f"Note: This reflects an observed statistical association and does not imply direct causation."
        )
        
    return {
        "r": round(r_val, 3),
        "p_value": round(p_val, 4),
        "strength": strength,
        "direction": direction,
        "interpretation": interp
    }


def compute_subject_statistics(df: pd.DataFrame, subject: str) -> Dict[str, any]:
    """
    Computes key summary statistics for a given subject.
    """
    sub_df = df[df["subject"] == subject]
    if sub_df.empty:
        return {}
        
    total_students = len(sub_df)
    passed_students = len(sub_df[sub_df["subject_result"] == "PASS"])
    pass_pct = (passed_students / total_students) * 100.0 if total_students > 0 else 0.0
    
    marks = sub_df["total_marks"]
    return {
        "subject": subject,
        "total_students": total_students,
        "average_marks": round(float(marks.mean()), 2),
        "highest_marks": round(float(marks.max()), 2),
        "lowest_marks": round(float(marks.min()), 2),
        "pass_percentage": round(float(pass_pct), 2),
        "std_dev": round(float(marks.std()), 2),
        "median_marks": round(float(marks.median()), 2)
    }
