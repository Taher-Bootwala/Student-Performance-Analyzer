"""
Rule-Based Insights Engine for Student Performance Analysis System.
Programmatically generates academic interpretations, strength/weakness evaluations,
and actionable recommendations.
"""

from typing import Dict, List
import pandas as pd


def generate_student_insights(student_row: pd.Series, student_subjects_df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generates tailored, rule-based academic insights for an individual student.
    Returns a list of dicts with 'status' (success, warning, info, danger), 'icon', and 'text'.
    """
    insights = []
    
    # 1. Attendance Evaluation
    att = float(student_row.get("attendance", 0))
    if att < 75.0:
        insights.append({
            "type": "warning",
            "label": "Attendance Alert",
            "text": f"Attendance is below the recommended 75% threshold ({att:.1f}%). Regular attendance strongly correlates with academic success."
        })
    elif att >= 85.0:
        insights.append({
            "type": "success",
            "label": "Attendance Record",
            "text": f"Excellent attendance record ({att:.1f}%). Demonstrates high engagement and discipline."
        })
    else:
        insights.append({
            "type": "success",
            "label": "Attendance Record",
            "text": f"Attendance is satisfactory ({att:.1f}%), meeting the institutional requirement."
        })

    # 2. Overall Academic Performance Evaluation
    pct = float(student_row.get("overall_percentage", 0))
    failed_count = int(student_row.get("failed_subjects", 0))
    
    if failed_count > 0:
        insights.append({
            "type": "danger",
            "label": "Remedial Alert",
            "text": f"Failed in {failed_count} subject(s). Urgent academic intervention and remedial support required."
        })
    elif pct >= 80.0:
        insights.append({
            "type": "success",
            "label": "Academic Excellence",
            "text": f"The student is performing very well academically ({pct:.1f}%, Grade: {student_row.get('grade', 'A')})."
        })
    elif pct >= 65.0:
        insights.append({
            "type": "success",
            "label": "Consistent Standing",
            "text": f"Overall performance is above average ({pct:.1f}%), maintaining consistent grades."
        })
    elif pct >= 50.0:
        insights.append({
            "type": "info",
            "label": "Average Performance",
            "text": f"Overall performance is average ({pct:.1f}%). Targeted revision can help elevate grades to distinction."
        })
    else:
        insights.append({
            "type": "warning",
            "label": "Borderline Standing",
            "text": f"Performance is borderline ({pct:.1f}%). Dedicated study plans and faculty guidance are advised."
        })

    # 3. Subject-Specific Discrepancy & Anomaly Detection
    if not student_subjects_df.empty:
        avg_student_marks = student_subjects_df["total_marks"].mean()
        
        # Weakest and strongest subjects
        min_row = student_subjects_df.loc[student_subjects_df["total_marks"].idxmin()]
        max_row = student_subjects_df.loc[student_subjects_df["total_marks"].idxmax()]
        
        # Check if weakest subject is significantly below student average
        if avg_student_marks - min_row["total_marks"] >= 12.0:
            insights.append({
                "type": "warning",
                "label": "Subject Support",
                "text": f"{min_row['subject']} requires improvement (Score: {min_row['total_marks']:.0f}/100 vs student average of {avg_student_marks:.1f}). This subject may require additional attention."
            })
            
        # Highlight standout subject
        if max_row["total_marks"] - avg_student_marks >= 10.0 and max_row["total_marks"] >= 75:
            insights.append({
                "type": "success",
                "label": "Subject Aptitude",
                "text": f"Strong aptitude observed in {max_row['subject']} (Score: {max_row['total_marks']:.0f}/100)."
            })

        # 4. Practical vs Theory Balance Check
        practical_pct = (student_subjects_df["practical_marks"].sum() / (len(student_subjects_df) * 20.0)) * 100.0
        final_theory_pct = (student_subjects_df["final_marks"].sum() / (len(student_subjects_df) * 50.0)) * 100.0
        
        if practical_pct - final_theory_pct >= 18.0:
            insights.append({
                "type": "info",
                "label": "Practical vs Theory",
                "text": f"The student demonstrates stronger practical performance ({practical_pct:.1f}%) than theoretical examination performance ({final_theory_pct:.1f}%). Strengthening conceptual answer writing is recommended."
            })
        elif final_theory_pct - practical_pct >= 18.0:
            insights.append({
                "type": "info",
                "label": "Practical vs Theory",
                "text": f"Strong theoretical performance ({final_theory_pct:.1f}%), but practical lab scores ({practical_pct:.1f}%) have scope for enhancement."
            })
        else:
            insights.append({
                "type": "success",
                "label": "Curricular Balance",
                "text": "Balanced performance between practical laboratories and theoretical examinations."
            })

        # 5. Continuous Assessment & Assignments
        assignment_avg = student_subjects_df["assignment_marks"].mean()
        if assignment_avg >= 8.5:
            insights.append({
                "type": "success",
                "label": "Coursework",
                "text": f"Consistently high assignment submission quality (Average: {assignment_avg:.1f}/10)."
            })
        elif assignment_avg < 6.0:
            insights.append({
                "type": "warning",
                "label": "Coursework",
                "text": f"Assignment marks are relatively low (Average: {assignment_avg:.1f}/10). Timely submissions can boost aggregate percentage."
            })

    return insights


def generate_class_level_insights(df: pd.DataFrame, summary_df: pd.DataFrame) -> List[str]:
    """
    Generates strategic batch-level insights for the entire cohort.
    """
    insights = []
    total_students = len(summary_df)
    pass_pct = (len(summary_df[summary_df["overall_result"] == "PASS"]) / total_students) * 100 if total_students > 0 else 0
    avg_att = summary_df["attendance"].mean()
    
    # Pass rate insight
    if pass_pct >= 85:
        insights.append(f"Outstanding cohort success rate: {pass_pct:.1f}% of students achieved a passing result across all subjects.")
    elif pass_pct >= 70:
        insights.append(f"Healthy cohort pass rate of {pass_pct:.1f}%. Mentoring targeted at borderline students can improve overall standing.")
    else:
        insights.append(f"Cohort pass rate stands at {pass_pct:.1f}%. Immediate curriculum review or remedial classes recommended.")

    # Attendance insight
    if avg_att < 75:
        insights.append(f"Cohort average attendance is low ({avg_att:.1f}%). Institution-level attendance enforcement could improve performance.")
    else:
        insights.append(f"Cohort attendance is strong at {avg_att:.1f}%, indicating healthy student engagement.")

    # Hardest and easiest subject
    subj_means = df.groupby("subject")["total_marks"].mean()
    hardest = subj_means.idxmin()
    easiest = subj_means.idxmax()
    insights.append(f"Subject Analysis: '{hardest}' had the lowest average score ({subj_means[hardest]:.1f}/100), while '{easiest}' saw the highest average ({subj_means[easiest]:.1f}/100).")

    return insights
