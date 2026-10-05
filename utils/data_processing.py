"""
Data Processing Module for Student Performance Analysis System.
Provides data loading, validation, grading, classification, and cleaning utilities.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# ==========================================
# 1. Configurable Grading & Thresholds
# ==========================================
GRADE_THRESHOLDS: List[Tuple[float, str]] = [
    (90.0, "A+"),
    (80.0, "A"),
    (70.0, "B+"),
    (60.0, "B"),
    (50.0, "C"),
    (40.0, "D"),
    (0.0, "F")
]

PASS_MARK_THRESHOLD: float = 40.0

CLASSIFICATION_THRESHOLDS: List[Tuple[float, str]] = [
    (80.0, "Excellent"),
    (65.0, "Good"),
    (50.0, "Average"),
    (0.0, "Needs Improvement")
]


def calculate_grade(score: float) -> str:
    """
    Returns letter grade based on configurable thresholds.
    """
    if pd.isna(score):
        return "N/A"
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


def classify_performance(score: float) -> str:
    """
    Classifies academic performance into qualitative categories:
    Excellent, Good, Average, Needs Improvement.
    """
    if pd.isna(score):
        return "N/A"
    for threshold, category in CLASSIFICATION_THRESHOLDS:
        if score >= threshold:
            return category
    return "Needs Improvement"


# ==========================================
# 2. Dataset Loading & Validation
# ==========================================
REQUIRED_CORE_COLUMNS = [
    "student_id",
    "student_name",
    "subject",
    "attendance",
    "internal_marks",
    "assignment_marks",
    "practical_marks",
    "final_marks"
]

COLUMN_SYNONYMS = {
    "id": "student_id",
    "roll_no": "student_id",
    "rollno": "student_id",
    "name": "student_name",
    "course": "subject",
    "att": "attendance",
    "internal": "internal_marks",
    "assignment": "assignment_marks",
    "practical": "practical_marks",
    "final": "final_marks",
    "final_exam": "final_marks"
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to lowercase snake_case and resolves synonyms."""
    new_cols = {}
    for col in df.columns:
        norm = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if norm in COLUMN_SYNONYMS:
            norm = COLUMN_SYNONYMS[norm]
        new_cols[col] = norm
    return df.rename(columns=new_cols)


def adapt_kaggle_dataset_if_detected(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Checks if the uploaded dataset is the UCI/Kaggle Student Performance dataset
    (G1, G2, G3, absences, school, sex, etc.) and converts it into the system schema.
    """
    cols = [c.lower() for c in df.columns]
    if "g1" in cols and "g2" in cols and "g3" in cols:
        records = []
        for idx, row in df.iterrows():
            sid = f"ST{1000 + idx + 1}"
            sname = f"Student {sid}"
            gender = "Male" if str(row.get("sex", "M")).upper() == "M" else "Female"
            dept = "General Studies"
            sem = "Semester 5"
            
            # Absences converted to attendance percentage (max ~93 days standard)
            absences = float(row.get("absences", 5))
            att_pct = max(30.0, min(100.0, 100.0 - (absences * 2.2)))
            
            # G1 is internal/midterm (out of 20)
            g1 = float(row.get("G1", 10))
            # G2 is practical/assignment (out of 20) -> split to 10 assignment, 10 practical
            g2 = float(row.get("G2", 10))
            # G3 is final exam (out of 20) -> scaled to 50
            g3 = float(row.get("G3", 10))
            
            # Map into 3 core subjects for variety
            subj_list = ["Mathematics", "Computer Science", "Applied Statistics"]
            for s_idx, subj in enumerate(subj_list):
                # slight deterministic variation across subjects
                delta = ((idx + s_idx) % 5) - 2
                internal = np.clip(round(g1 + delta * 0.5), 0, 20)
                assignment = np.clip(round((g2 / 2.0) + delta * 0.2), 0, 10)
                practical = np.clip(round((g2 / 2.0) + delta * 0.3), 0, 20)
                final = np.clip(round((g3 * 2.5) + delta * 1.5), 0, 50)
                sub_att = np.clip(round(att_pct + delta), 40, 100)
                
                records.append({
                    "student_id": sid,
                    "student_name": sname,
                    "gender": gender,
                    "department": dept,
                    "semester": sem,
                    "subject": subj,
                    "attendance": round(sub_att, 1),
                    "internal_marks": int(internal),
                    "assignment_marks": int(assignment),
                    "practical_marks": int(practical),
                    "final_marks": int(final)
                })
        return pd.DataFrame(records)
    return None


def validate_and_prepare_dataset(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Validates the dataset columns, handles variations or adapts Kaggle format,
    and returns a clean, structured DataFrame with total_marks computed.
    """
    if df is None or df.empty:
        return None, "The uploaded dataset is empty."
    
    df = normalize_column_names(df.copy())
    
    # Check if this is a Kaggle/UCI student dataset
    adapted = adapt_kaggle_dataset_if_detected(df)
    if adapted is not None:
        df = adapted
        
    missing_cols = [c for c in REQUIRED_CORE_COLUMNS if c not in df.columns]
    
    # Special handling if total_marks is already given but individual components are missing
    if "total_marks" in df.columns and "subject" in df.columns and "attendance" in df.columns:
        if "internal_marks" not in df.columns:
            df["internal_marks"] = (df["total_marks"] * 0.20).round().astype(int)
        if "assignment_marks" not in df.columns:
            df["assignment_marks"] = (df["total_marks"] * 0.10).round().astype(int)
        if "practical_marks" not in df.columns:
            df["practical_marks"] = (df["total_marks"] * 0.20).round().astype(int)
        if "final_marks" not in df.columns:
            df["final_marks"] = (df["total_marks"] - df["internal_marks"] - df["assignment_marks"] - df["practical_marks"]).clip(lower=0).astype(int)
        missing_cols = [c for c in REQUIRED_CORE_COLUMNS if c not in df.columns]

    if missing_cols:
        return None, f"Dataset is missing required columns: {', '.join(missing_cols)}"

    # Ensure optional descriptive columns exist
    if "gender" not in df.columns:
        df["gender"] = "All"
    if "department" not in df.columns:
        df["department"] = "Computer Engineering"
    if "semester" not in df.columns:
        df["semester"] = "Semester 5"

    # Ensure numeric columns are properly typed
    numeric_cols = ["attendance", "internal_marks", "assignment_marks", "practical_marks", "final_marks"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute Subject Total Marks (out of 100)
    df["total_marks"] = (
        df["internal_marks"].fillna(0) +
        df["assignment_marks"].fillna(0) +
        df["practical_marks"].fillna(0) +
        df["final_marks"].fillna(0)
    )
    
    # Subject-level percentage, grade and result
    df["subject_percentage"] = df["total_marks"].clip(0, 100)
    df["subject_grade"] = df["subject_percentage"].apply(calculate_grade)
    df["subject_result"] = np.where(df["total_marks"] >= PASS_MARK_THRESHOLD, "PASS", "FAIL")

    return df, None


# ==========================================
# 3. Student-Level Aggregation & Ranking
# ==========================================
def compute_student_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates subject-level records to student-level summary.
    Computes overall percentage, grade, pass/fail result, and ranking.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    def aggregate_student(group: pd.DataFrame) -> pd.Series:
        total_subjects = len(group)
        sum_total_marks = group["total_marks"].sum()
        max_possible_marks = total_subjects * 100.0 if total_subjects > 0 else 100.0
        overall_percentage = (sum_total_marks / max_possible_marks) * 100.0
        
        # Result: Student passes only if they pass all individual subjects AND overall >= 40
        failed_subjects = (group["total_marks"] < PASS_MARK_THRESHOLD).sum()
        overall_result = "PASS" if (failed_subjects == 0 and overall_percentage >= PASS_MARK_THRESHOLD) else "FAIL"
        
        avg_attendance = group["attendance"].mean()
        
        return pd.Series({
            "student_name": group["student_name"].iloc[0],
            "gender": group["gender"].iloc[0] if "gender" in group.columns else "N/A",
            "department": group["department"].iloc[0] if "department" in group.columns else "N/A",
            "semester": group["semester"].iloc[0] if "semester" in group.columns else "N/A",
            "total_subjects": total_subjects,
            "overall_marks": round(float(group["total_marks"].mean()), 2),
            "overall_percentage": round(float(overall_percentage), 2),
            "attendance": round(float(avg_attendance), 2),
            "failed_subjects": failed_subjects,
            "overall_result": overall_result,
            "grade": calculate_grade(overall_percentage),
            "classification": classify_performance(overall_percentage)
        })

    summary = df.groupby("student_id", as_index=False).apply(aggregate_student)
    # Reset index if pandas creates multi-index
    if isinstance(summary.index, pd.MultiIndex):
        summary = summary.reset_index(level=1, drop=True).reset_index()

    # Sort descending by overall percentage and assign ranks
    summary = summary.sort_values(by="overall_percentage", ascending=False).reset_index(drop=True)
    summary["rank"] = summary["overall_percentage"].rank(ascending=False, method="min").astype(int)
    
    return summary


# ==========================================
# 4. Data Cleaning Utilities
# ==========================================
def get_missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame detailing missing values count and percentage."""
    missing = df.isnull().sum()
    pct = (missing / len(df)) * 100
    report = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing.values,
        "Percentage (%)": pct.round(2).values
    })
    return report[report["Missing Values"] > 0].reset_index(drop=True)


def get_duplicate_rows_count(df: pd.DataFrame) -> int:
    """Returns number of duplicate rows in the DataFrame."""
    return int(df.duplicated().sum())


def clean_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """
    Cleans missing values by either dropping rows or filling numeric cols with mean/median.
    """
    df_clean = df.copy()
    if strategy == "drop":
        df_clean = df_clean.dropna().reset_index(drop=True)
    elif strategy == "fill_mean":
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            mean_val = df_clean[col].mean()
            df_clean[col] = df_clean[col].fillna(mean_val)
    elif strategy == "fill_median":
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
    # Recompute total_marks and subject percentage after cleaning
    df_clean, _ = validate_and_prepare_dataset(df_clean)
    return df_clean


def clean_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate records from DataFrame."""
    df_clean = df.drop_duplicates().reset_index(drop=True)
    df_clean, _ = validate_and_prepare_dataset(df_clean)
    return df_clean
