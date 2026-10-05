"""
Student Performance Analysis System
Python for Data Science Micro-Project
Built with Streamlit, Pandas, NumPy, SciPy, Matplotlib, and Seaborn.
Strict Flat-Color Academic Design (NO GRADIENTS).
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Import custom modular utilities and charts
from utils.data_processing import (
    validate_and_prepare_dataset,
    compute_student_summary,
    get_missing_values_report,
    get_duplicate_rows_count,
    clean_missing_values,
    clean_duplicate_records,
    PASS_MARK_THRESHOLD
)
from utils.statistics import (
    compute_descriptive_stats,
    generate_numerical_summary_table,
    compute_correlation_with_interpretation,
    compute_subject_statistics
)
from utils.insights import (
    generate_student_insights,
    generate_class_level_insights
)
from charts.visualizations import (
    plot_pass_fail_donut,
    plot_subject_averages,
    plot_attendance_vs_marks,
    plot_grade_distribution,
    plot_top_students,
    plot_student_subject_breakdown,
    plot_subject_distribution,
    plot_subject_boxplot,
    plot_correlation_heatmap,
    plot_scatter_trend
)

# ==============================================================================
# 1. Page Configuration & Theme Loading
# ==============================================================================
st.set_page_config(
    page_title="Student Performance Analysis System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS Stylesheet
STYLE_PATH = Path("assets/style.css")
if STYLE_PATH.exists():
    with open(STYLE_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

DEFAULT_DATA_PATH = Path("data/students.csv")
DIRTY_DATA_PATH = Path("data/students_raw_uncleaned.csv")


# ==============================================================================
# 2. State Initialization & Data Loading
# ==============================================================================
def load_initial_dataset():
    """Loads the default dataset or uncleaned demo dataset."""
    if DEFAULT_DATA_PATH.exists():
        raw = pd.read_csv(DEFAULT_DATA_PATH)
    else:
        st.error("Default dataset not found at data/students.csv.")
        raw = pd.DataFrame()
    return raw


if "raw_df" not in st.session_state or st.session_state["raw_df"] is None:
    st.session_state["raw_df"] = load_initial_dataset()
    st.session_state["dataset_source"] = "Default Sample Dataset (120 Students, 5 Subjects)"

# Validate and process active dataset
processed_df, error_msg = validate_and_prepare_dataset(st.session_state["raw_df"])
if error_msg:
    st.session_state["processed_df"] = None
    st.session_state["student_summary"] = None
    st.session_state["validation_error"] = error_msg
else:
    st.session_state["processed_df"] = processed_df
    st.session_state["student_summary"] = compute_student_summary(processed_df)
    st.session_state["validation_error"] = None


# ==============================================================================
# 3. Sidebar Navigation & Global Controls
# ==============================================================================
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0 18px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 15px;">
            <div style="font-size: 1.25rem; font-weight: 800; color: #1E3A5F; line-height: 1.2;">
                Student Performance Analyzer
            </div>
            <div style="font-size: 0.80rem; color: #64748B; margin-top: 4px;">
                Python for Data Science Micro-Project
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Navigation Menu
    nav_selection = st.radio(
        "Navigation",
        [
            "1. Dashboard",
            "2. Student Analysis",
            "3. Subject Analysis",
            "4. Performance Trends",
            "5. Dataset",
            "6. About Project"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #1E3A5F; text-transform: uppercase; margin-bottom: 8px;'>Dataset Management</div>", unsafe_allow_html=True)
    
    # Dataset File Uploader
    uploaded_file = st.file_uploader(
        "Upload Custom CSV",
        type=["csv"],
        help="Upload an academic CSV with student marks or Kaggle student performance data."
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            test_proc, test_err = validate_and_prepare_dataset(uploaded_df)
            if test_err:
                st.error(f"Invalid dataset: {test_err}")
            else:
                st.session_state["raw_df"] = uploaded_df
                st.session_state["dataset_source"] = f"Uploaded: {uploaded_file.name}"
                st.session_state["processed_df"] = test_proc
                st.session_state["student_summary"] = compute_student_summary(test_proc)
                st.session_state["validation_error"] = None
                st.success("Dataset uploaded and processed successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Error parsing CSV file: {str(e)}")

    # Reset Button
    col_reset, col_demo = st.columns(2)
    with col_reset:
        if st.button("Reset Default", use_container_width=True):
            st.session_state["raw_df"] = load_initial_dataset()
            st.session_state["dataset_source"] = "Default Sample Dataset (120 Students, 5 Subjects)"
            st.rerun()
            
    with col_demo:
        if st.button("Load Dirty CSV", use_container_width=True, help="Loads dataset with missing values & duplicates to test data cleaning"):
            if DIRTY_DATA_PATH.exists():
                st.session_state["raw_df"] = pd.read_csv(DIRTY_DATA_PATH)
                st.session_state["dataset_source"] = "Uncleaned Demo Dataset (with missing values & duplicates)"
                st.rerun()

    # Active Dataset Status Info
    st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px; margin-top: 12px; font-size: 0.78rem; color: #64748B;">
            <b>Active:</b> {st.session_state.get('dataset_source', 'Default')}<br>
            <b>Records:</b> {len(st.session_state['raw_df'])} rows
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #1E3A5F; text-transform: uppercase; margin-bottom: 8px;'>Cohort Filters</div>", unsafe_allow_html=True)

    # Global Filters applied to the views
    current_df = st.session_state["processed_df"]
    filter_dept = "All"
    filter_sem = "All"
    filter_gender = "All"
    filter_result = "All"

    if current_df is not None and not current_df.empty:
        depts = ["All"] + sorted([str(d) for d in current_df["department"].dropna().unique()])
        filter_dept = st.selectbox("Department / Stream", depts)

        if "gender" in current_df.columns:
            genders = ["All"] + sorted([str(g) for g in current_df["gender"].dropna().unique()])
            filter_gender = st.selectbox("Gender", genders)

        filter_result = st.selectbox("Result Status", ["All", "PASS", "FAIL"])


# Helper function to apply global filters
def get_filtered_data():
    df = st.session_state["processed_df"]
    summary = st.session_state["student_summary"]
    
    if df is None or summary is None or df.empty:
        return df, summary
        
    filtered_summary = summary.copy()
    
    if filter_dept != "All":
        filtered_summary = filtered_summary[filtered_summary["department"] == filter_dept]
    if filter_gender != "All" and "gender" in filtered_summary.columns:
        filtered_summary = filtered_summary[filtered_summary["gender"] == filter_gender]
    if filter_result != "All":
        filtered_summary = filtered_summary[filtered_summary["overall_result"] == filter_result]

    valid_student_ids = filtered_summary["student_id"].unique()
    filtered_df = df[df["student_id"].isin(valid_student_ids)].copy()
    
    return filtered_df, filtered_summary


active_df, active_summary = get_filtered_data()


# ==============================================================================
# Check for validation error before rendering pages
# ==============================================================================
if st.session_state.get("validation_error"):
    st.error(f"### Dataset Error\n\n{st.session_state['validation_error']}")
    st.info("Please upload a valid CSV file with student assessment columns or click **Reset Default** in the sidebar.")
    st.stop()


# ==============================================================================
# PAGE 1: DASHBOARD PAGE
# ==============================================================================
if nav_selection.startswith("1."):

    if active_summary.empty:
        st.warning("No student records match the selected sidebar filters.")
        st.stop()

    # KPI Calculation
    total_students = len(active_summary)
    avg_attendance = active_summary["attendance"].mean()
    avg_marks = active_summary["overall_marks"].mean()
    passed_count = len(active_summary[active_summary["overall_result"] == "PASS"])
    pass_pct = (passed_count / total_students * 100.0) if total_students > 0 else 0.0
    top_row = active_summary.iloc[0]
    top_student_text = f"{top_row['student_name']}"
    top_student_sub = f"{top_row['overall_percentage']:.1f}% ({top_row['student_id']})"

    # ── Hero Banner with animated stats strip ──
    st.markdown(f"""
        <div class="hero-banner">
            <span class="hero-tag">Academic Analytics System</span>
            <div class="hero-title">Student Performance Dashboard</div>
            <div class="hero-subtitle">Comprehensive academic analytics powered by Python, Pandas, SciPy, Matplotlib &amp; Seaborn</div>
            <div class="hero-stats-strip">
                <div class="hero-stat-chip">
                    <span class="chip-label">Students</span>
                    <span class="chip-value">{total_students}</span>
                </div>
                <div class="hero-stat-chip">
                    <span class="chip-label">Attendance</span>
                    <span class="chip-value">{avg_attendance:.1f}%</span>
                </div>
                <div class="hero-stat-chip">
                    <span class="chip-label">Pass Rate</span>
                    <span class="chip-value">{pass_pct:.1f}%</span>
                </div>
                <div class="hero-stat-chip">
                    <span class="chip-label">Top Scorer</span>
                    <span class="chip-value">{top_row['overall_percentage']:.1f}%</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards with icons removed, stagger, and progress bars ──
    att_color = "fill-success" if avg_attendance >= 75 else "fill-warning"
    pass_color = "fill-success" if pass_pct >= 70 else ("fill-warning" if pass_pct >= 50 else "fill-danger")
    marks_color = "fill-primary" if avg_marks >= 60 else "fill-warning"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-1">
                <div class="kpi-title">Total Students</div>
                <div class="kpi-value">{total_students}</div>
                <div class="kpi-sub">Cohort Enrolled</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-2">
                <div class="kpi-title">Avg Attendance</div>
                <div class="kpi-value">{avg_attendance:.1f}%</div>
                <div class="kpi-sub">Target: &gt;= 75%</div>
                <div class="progress-track"><div class="progress-fill {att_color}" style="--fill-width: {min(avg_attendance, 100):.0f}%;"></div></div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-3">
                <div class="kpi-title">Average Marks</div>
                <div class="kpi-value">{avg_marks:.1f}</div>
                <div class="kpi-sub">Out of 100</div>
                <div class="progress-track"><div class="progress-fill {marks_color}" style="--fill-width: {min(avg_marks, 100):.0f}%;"></div></div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-4">
                <div class="kpi-title">Pass Percentage</div>
                <div class="kpi-value">{pass_pct:.1f}%</div>
                <div class="kpi-sub">{passed_count} of {total_students} Students</div>
                <div class="progress-track"><div class="progress-fill {pass_color}" style="--fill-width: {min(pass_pct, 100):.0f}%;"></div></div>
            </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-5 kpi-highlight">
                <div class="kpi-title">Top Performer</div>
                <div class="kpi-value" style="font-size: 1.25rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{top_student_text}</div>
                <div class="kpi-sub">{top_student_sub}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Visualizations Row 1: Pass/Fail and Subject Averages
    col_chart1, col_chart2 = st.columns([1, 1.2])
    with col_chart1:
        st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
        with st.spinner("Rendering chart..."):
            fig1 = plot_pass_fail_donut(active_summary)
            st.pyplot(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_chart2:
        st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
        with st.spinner("Rendering chart..."):
            fig2 = plot_subject_averages(active_df)
            st.pyplot(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Visualizations Row 2: Attendance vs Overall Marks & Grade Distribution
    col_chart3, col_chart4 = st.columns([1.1, 1])
    with col_chart3:
        st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
        with st.spinner("Rendering chart..."):
            fig3 = plot_attendance_vs_marks(active_summary)
            st.pyplot(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_chart4:
        st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
        with st.spinner("Rendering chart..."):
            fig4 = plot_grade_distribution(active_summary)
            st.pyplot(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Visualizations Row 3: Top 10 Students & Cohort Strategic Insights
    col_chart5, col_insights = st.columns([1.2, 1])
    with col_chart5:
        st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
        with st.spinner("Rendering chart..."):
            fig5 = plot_top_students(active_summary, top_n=min(10, len(active_summary)))
            st.pyplot(fig5, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_insights:
        st.markdown("""
            <div class="content-card" style="height: 100%;">
                <div style="font-size: 1.08rem; font-weight: 800; color: #1E3A5F; margin-bottom: 14px;">
                    Strategic Cohort Observations
                </div>
        """, unsafe_allow_html=True)
        
        cohort_insights = generate_class_level_insights(active_df, active_summary)
        for ins in cohort_insights:
            st.markdown(f"""
                <div class="insight-card insight-info">
                    <div>{ins}</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: STUDENT ANALYSIS PAGE
# ==============================================================================
elif nav_selection.startswith("2."):
    st.markdown("""
        <div class="section-banner">
            <h2>Individual Student Performance Analysis</h2>
            <p>Detailed profile assessment, marks breakdown, and rule-based diagnostic summary for individual students.</p>
        </div>
    """, unsafe_allow_html=True)

    if active_summary.empty:
        st.warning("No student records available.")
        st.stop()

    # Search & Selection by Student ID or Name
    student_options = [
        f"{row['student_id']} - {row['student_name']}" for _, row in active_summary.iterrows()
    ]
    
    col_sel, col_quick = st.columns([2, 1])
    with col_sel:
        selected_option = st.selectbox("Search / Select Student (by ID or Name):", student_options)
    
    selected_id = selected_option.split(" - ")[0].strip()
    
    # Retrieve student record and subject breakdown
    student_row = active_summary[active_summary["student_id"] == selected_id].iloc[0]
    student_subjects = active_df[active_df["student_id"] == selected_id].copy()

    # Format result and classification badges
    res_badge = "badge-pass" if student_row["overall_result"] == "PASS" else "badge-fail"
    cls_map = {
        "Excellent": "badge-excellent",
        "Good": "badge-good",
        "Average": "badge-average",
        "Needs Improvement": "badge-warning"
    }
    cls_badge = cls_map.get(student_row["classification"], "badge-good")

    # Student Profile Card
    st.markdown(f"""
        <div class="student-profile-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
                <div>
                    <h3 class="profile-name">{student_row['student_name']}</h3>
                    <div class="profile-meta">
                        <b>Student ID:</b> {student_row['student_id']} &nbsp;|&nbsp; 
                        <b>Department:</b> {student_row['department']} &nbsp;|&nbsp; 
                        <b>Semester:</b> {student_row['semester']} &nbsp;|&nbsp; 
                        <b>Gender:</b> {student_row['gender']}
                    </div>
                </div>
                <div>
                    <span class="badge {res_badge}" style="font-size: 0.95rem; padding: 6px 14px;">{student_row['overall_result']}</span>
                    &nbsp;
                    <span class="badge {cls_badge}" style="font-size: 0.95rem; padding: 6px 14px;">{student_row['classification']}</span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 15px; border-top: 1px solid #E2E8F0; padding-top: 15px;">
                <div>
                    <div class="kpi-title">Overall Percentage</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #1E3A5F;">{student_row['overall_percentage']:.1f}%</div>
                </div>
                <div>
                    <div class="kpi-title">Grade Awarded</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #1E3A5F;">{student_row['grade']}</div>
                </div>
                <div>
                    <div class="kpi-title">Cohort Rank</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #1E3A5F;">#{student_row['rank']} <span style="font-size: 0.8rem; color: #64748B;">of {len(st.session_state['student_summary'])}</span></div>
                </div>
                <div>
                    <div class="kpi-title">Attendance</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #1E3A5F;">{student_row['attendance']:.1f}%</div>
                </div>
                <div>
                    <div class="kpi-title">Failed Subjects</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: {'#DC2626' if student_row['failed_subjects'] > 0 else '#16A34A'};">{student_row['failed_subjects']}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Subject-wise marks breakdown and Visualizations
    col_table, col_chart = st.columns([1.1, 1.2])

    with col_table:
        st.markdown("<div class='content-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #1E3A5F; margin-bottom: 10px;'>Subject Assessment Breakdown</div>", unsafe_allow_html=True)
        display_sub_df = student_subjects[[
            "subject", "attendance", "internal_marks", "assignment_marks", "practical_marks", "final_marks", "total_marks", "subject_grade", "subject_result"
        ]].copy()
        display_sub_df.columns = ["Subject", "Att %", "Internal (20)", "Assign (10)", "Practical (20)", "Final (50)", "Total (100)", "Grade", "Status"]
        st.dataframe(display_sub_df, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart:
        st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
        with st.spinner("Rendering assessment radar/bars..."):
            fig_sub = plot_student_subject_breakdown(student_subjects)
            st.pyplot(fig_sub, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Programmatic Performance Summary & Insights
    st.markdown("<div style='font-size: 1.1rem; font-weight: 700; color: #1E3A5F; margin: 18px 0 10px 0;'>Performance Diagnostic Summary & Recommendations</div>", unsafe_allow_html=True)
    
    insights = generate_student_insights(student_row, student_subjects)
    for ins in insights:
        card_class = f"insight-{ins['type']}"
        st.markdown(f"""
            <div class="insight-card {card_class}">
                <div style="flex: 1;">
                    <span class="insight-label">{ins.get('label', 'Diagnostic')}</span>
                    <div style="margin-top: 2px;">{ins['text']}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: SUBJECT ANALYSIS PAGE
# ==============================================================================
elif nav_selection.startswith("3."):
    st.markdown("""
        <div class="section-banner">
            <h2>Subject-Wise Performance Analysis</h2>
            <p>Explore curricular distributions, quartile boxplots, top performers, and identify students requiring remedial attention.</p>
        </div>
    """, unsafe_allow_html=True)

    if active_df.empty:
        st.warning("No records available.")
        st.stop()

    available_subjects = sorted(active_df["subject"].unique())
    selected_subject = st.selectbox("Select Subject for Detailed Analysis:", available_subjects)

    # Compute Subject Statistics
    sub_stats = compute_subject_statistics(active_df, selected_subject)

    # Subject KPI metrics
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-1">
                <div class="kpi-title">Average Marks</div>
                <div class="kpi-value">{sub_stats['average_marks']:.1f}</div>
                <div class="kpi-sub">Median: {sub_stats['median_marks']:.1f}</div>
            </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-2">
                <div class="kpi-title">Highest Marks</div>
                <div class="kpi-value">{sub_stats['highest_marks']:.0f}</div>
                <div class="kpi-sub">Out of 100</div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-3">
                <div class="kpi-title">Lowest Marks</div>
                <div class="kpi-value">{sub_stats['lowest_marks']:.0f}</div>
                <div class="kpi-sub">Pass Mark: {PASS_MARK_THRESHOLD:.0f}</div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-4">
                <div class="kpi-title">Pass Percentage</div>
                <div class="kpi-value">{sub_stats['pass_percentage']:.1f}%</div>
                <div class="kpi-sub">Subject Cleared</div>
                <div class="progress-track"><div class="progress-fill fill-success" style="--fill-width: {min(sub_stats['pass_percentage'], 100):.0f}%;"></div></div>
            </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-5">
                <div class="kpi-title">Enrolled Count</div>
                <div class="kpi-value">{sub_stats['total_students']}</div>
                <div class="kpi-sub">Active Students</div>
            </div>
        """, unsafe_allow_html=True)
    with k6:
        st.markdown(f"""
            <div class="kpi-container" style="animation-delay: 0.40s;">
                <div class="kpi-title">Std Deviation</div>
                <div class="kpi-value">{sub_stats['std_dev']:.1f}</div>
                <div class="kpi-sub">Score Spread</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Charts: Histogram and Boxplot
    col_hist, col_box = st.columns(2)
    with col_hist:
        st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
        with st.spinner("Rendering histogram..."):
            fig_dist = plot_subject_distribution(active_df, selected_subject)
            st.pyplot(fig_dist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_box:
        st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
        with st.spinner("Rendering boxplot..."):
            fig_box = plot_subject_boxplot(active_df, selected_subject)
            st.pyplot(fig_box, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Top Performers and Weak Performers Tables
    sub_records = active_df[active_df["subject"] == selected_subject].copy()

    col_top, col_weak = st.columns(2)
    with col_top:
        st.markdown(f"""
            <div class="content-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #1E3A5F; margin-bottom: 10px;">
                    Top 5 Performers in {selected_subject}
                </div>
        """, unsafe_allow_html=True)
        top5 = sub_records.sort_values(by="total_marks", ascending=False).head(5)
        top5_display = top5[["student_id", "student_name", "attendance", "total_marks", "subject_grade"]].copy()
        top5_display.columns = ["Student ID", "Student Name", "Att %", "Marks", "Grade"]
        st.dataframe(top5_display, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_weak:
        st.markdown(f"""
            <div class="content-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #1E3A5F; margin-bottom: 10px;">
                    Weak Performers / Remedial Watchlist
                </div>
        """, unsafe_allow_html=True)
        threshold = st.slider(f"Filter students scoring below threshold marks:", min_value=20, max_value=60, value=40, step=5)
        weak_df = sub_records[sub_records["total_marks"] < threshold].sort_values(by="total_marks", ascending=True)
        
        if weak_df.empty:
            st.success(f"No students scored below {threshold} marks in {selected_subject}!")
        else:
            st.warning(f"{len(weak_df)} student(s) identified with scores below {threshold} marks.")
            weak_display = weak_df[["student_id", "student_name", "attendance", "internal_marks", "final_marks", "total_marks", "subject_result"]].copy()
            weak_display.columns = ["Student ID", "Student Name", "Att %", "Internal", "Final Exam", "Total", "Status"]
            st.dataframe(weak_display, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: PERFORMANCE TRENDS PAGE
# ==============================================================================
elif nav_selection.startswith("4."):
    st.markdown("""
        <div class="section-banner">
            <h2>Academic Performance Trends & Correlation Analysis</h2>
            <p>Empirical investigation of relationships between attendance, continuous internal assessments, and terminal examinations.</p>
        </div>
    """, unsafe_allow_html=True)

    if active_df.empty:
        st.warning("No records available.")
        st.stop()

    trend_choice = st.radio(
        "Select Trend Investigation:",
        [
            "Attendance vs Marks",
            "Internal vs Final Exam",
            "Assignment vs Final Exam",
            "Correlation Heatmap"
        ],
        horizontal=True
    )
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if trend_choice == "Attendance vs Marks":
        col_c1, col_t1 = st.columns([1.2, 1])
        res1 = compute_correlation_with_interpretation(
            active_summary["attendance"],
            active_summary["overall_marks"],
            x_name="Attendance (%)",
            y_name="Overall Marks"
        )
        with col_c1:
            st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
            with st.spinner("Rendering scatter plot..."):
                fig_t1 = plot_scatter_trend(
                    active_summary,
                    "attendance",
                    "overall_marks",
                    "Attendance Percentage (%)",
                    "Overall Marks (out of 100)",
                    "Attendance % vs Overall Academic Marks",
                    r_val=res1["r"]
                )
                st.pyplot(fig_t1, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_t1:
            st.markdown(f"""
                <div class="stat-highlight">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #1E3A5F; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em;">Statistical Correlation</div>
                    <div class="stat-r">r = {res1['r']:.2f}</div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 14px;">p-value: {res1['p_value']:.4f} &bull; <b>{res1['strength']} {res1['direction']}</b></div>
                    <div class="insight-card insight-info" style="font-size: 0.90rem; margin-bottom: 12px;">
                        <div>{res1['interpretation']}</div>
                    </div>
                    <div style="font-size: 0.80rem; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 10px;">
                        <b>Methodology Note:</b> Calculated using SciPy Pearson correlation coefficient (r) on cohort averages.
                    </div>
                </div>
            """, unsafe_allow_html=True)

    elif trend_choice == "Internal vs Final Exam":
        col_c2, col_t2 = st.columns([1.2, 1])
        res2 = compute_correlation_with_interpretation(
            active_df["internal_marks"],
            active_df["final_marks"],
            x_name="Internal Examination Marks",
            y_name="Final Examination Marks"
        )
        with col_c2:
            st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
            with st.spinner("Rendering scatter plot..."):
                fig_t2 = plot_scatter_trend(
                    active_df,
                    "internal_marks",
                    "final_marks",
                    "Internal Exam Marks (out of 20)",
                    "Final Exam Marks (out of 50)",
                    "Internal Exam Marks vs Final Examination Marks",
                    r_val=res2["r"]
                )
                st.pyplot(fig_t2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_t2:
            st.markdown(f"""
                <div class="stat-highlight">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #1E3A5F; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em;">Statistical Correlation</div>
                    <div class="stat-r">r = {res2['r']:.2f}</div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 14px;">p-value: {res2['p_value']:.4f} &bull; <b>{res2['strength']} {res2['direction']}</b></div>
                    <div class="insight-card insight-info" style="font-size: 0.90rem; margin-bottom: 12px;">
                        <div>{res2['interpretation']}</div>
                    </div>
                    <div style="font-size: 0.80rem; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 10px;">
                        <b>Academic Significance:</b> Indicates how strongly continuous mid-term testing predicts summative university finals.
                    </div>
                </div>
            """, unsafe_allow_html=True)

    elif trend_choice == "Assignment vs Final Exam":
        col_c3, col_t3 = st.columns([1.2, 1])
        res3 = compute_correlation_with_interpretation(
            active_df["assignment_marks"],
            active_df["final_marks"],
            x_name="Assignment Marks",
            y_name="Final Examination Marks"
        )
        with col_c3:
            st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
            with st.spinner("Rendering scatter plot..."):
                fig_t3 = plot_scatter_trend(
                    active_df,
                    "assignment_marks",
                    "final_marks",
                    "Assignment Marks (out of 10)",
                    "Final Exam Marks (out of 50)",
                    "Assignment Marks vs Final Examination Marks",
                    r_val=res3["r"]
                )
                st.pyplot(fig_t3, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_t3:
            st.markdown(f"""
                <div class="stat-highlight">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #1E3A5F; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em;">Statistical Correlation</div>
                    <div class="stat-r">r = {res3['r']:.2f}</div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 14px;">p-value: {res3['p_value']:.4f} &bull; <b>{res3['strength']} {res3['direction']}</b></div>
                    <div class="insight-card insight-info" style="font-size: 0.90rem; margin-bottom: 12px;">
                        <div>{res3['interpretation']}</div>
                    </div>
                    <div style="font-size: 0.80rem; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 10px;">
                        <b>Academic Note:</b> Homework and coursework submissions serve as ongoing indicators of student diligence.
                    </div>
                </div>
            """, unsafe_allow_html=True)

    elif trend_choice == "Correlation Heatmap":
        col_heat, col_matrix = st.columns([1.1, 1])
        with col_heat:
            st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
            with st.spinner("Rendering correlation heatmap..."):
                fig_heat = plot_correlation_heatmap(active_df)
                st.pyplot(fig_heat, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_matrix:
            st.markdown("""
                <div class="content-card">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #1E3A5F; margin-bottom: 10px;">Numerical Correlation Matrix</div>
            """, unsafe_allow_html=True)
            target_cols = ["attendance", "internal_marks", "assignment_marks", "practical_marks", "final_marks", "total_marks"]
            valid_cols = [c for c in target_cols if c in active_df.columns]
            corr_df = active_df[valid_cols].corr().round(3)
            st.dataframe(corr_df, use_container_width=True)
            st.markdown("""
                    <div style="font-size: 0.82rem; color: #64748B; margin-top: 10px; line-height: 1.5;">
                        <b>Data Science Principle:</b> Values range between -1.0 (perfect inverse relationship) to +1.0 (perfect direct relationship). 
                        A correlation coefficient does not establish a causal mechanism; rather, it highlights linear co-variation across variables.
                    </div>
                </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 5: DATASET & DATA CLEANING PAGE
# ==============================================================================
elif nav_selection.startswith("5."):
    st.markdown("""
        <div class="section-banner">
            <h2>Dataset Management & Data Cleaning Pipeline</h2>
            <p>Inspect raw records, audit missing values, detect duplicates, and execute in-memory cleaning operations.</p>
        </div>
    """, unsafe_allow_html=True)

    raw_data = st.session_state["raw_df"]
    
    # Metadata Overview KPIs
    m1, m2, m3, m4 = st.columns(4)
    missing_report = get_missing_values_report(raw_data)
    total_missing_cells = missing_report["Missing Values"].sum() if not missing_report.empty else 0
    duplicate_rows = get_duplicate_rows_count(raw_data)

    with m1:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-1">
                <div class="kpi-title">Total Rows</div>
                <div class="kpi-value">{len(raw_data)}</div>
                <div class="kpi-sub">Dataset Records</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-2">
                <div class="kpi-title">Total Columns</div>
                <div class="kpi-value">{len(raw_data.columns)}</div>
                <div class="kpi-sub">Features & Target</div>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-3">
                <div class="kpi-title">Missing Values</div>
                <div class="kpi-value" style="color: {'#DC2626' if total_missing_cells > 0 else '#16A34A'};">{total_missing_cells}</div>
                <div class="kpi-sub">{len(missing_report)} affected column(s)</div>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
            <div class="kpi-container kpi-delay-4">
                <div class="kpi-title">Duplicate Rows</div>
                <div class="kpi-value" style="color: {'#DC2626' if duplicate_rows > 0 else '#16A34A'};">{duplicate_rows}</div>
                <div class="kpi-sub">Exact duplicate tuples</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Interactive Table Viewer
    st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #1E3A5F; margin-bottom: 8px;'>Active Dataset View</div>", unsafe_allow_html=True)
    st.dataframe(raw_data, use_container_width=True)

    # Data Cleaning Operations Section
    st.markdown("---")
    st.markdown("<div style='font-size: 1.15rem; font-weight: 700; color: #1E3A5F; margin-bottom: 12px;'>Data Cleaning & Preprocessing Controls</div>", unsafe_allow_html=True)

    col_clean_miss, col_clean_dup = st.columns(2)

    with col_clean_miss:
        st.markdown("""
            <div class="content-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #1E3A5F; margin-bottom: 10px;">
                    1. Handle Missing Values
                </div>
        """, unsafe_allow_html=True)
        
        if total_missing_cells == 0:
            st.success("No missing values detected in the current dataset.")
        else:
            st.warning(f"Detected {total_missing_cells} missing values across features:")
            st.dataframe(missing_report, hide_index=True, use_container_width=True)
            
            clean_strategy = st.radio(
                "Select Imputation Strategy:",
                [
                    ("Drop rows containing null values", "drop"),
                    ("Impute numeric missing values with Mean", "fill_mean"),
                    ("Impute numeric missing values with Median", "fill_median")
                ],
                format_func=lambda x: x[0]
            )
            
            if st.button("Apply Missing Value Treatment", use_container_width=True):
                cleaned = clean_missing_values(raw_data, strategy=clean_strategy[1])
                st.session_state["raw_df"] = cleaned
                st.success("Missing value treatment applied successfully!")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_clean_dup:
        st.markdown("""
            <div class="content-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #1E3A5F; margin-bottom: 10px;">
                    2. Handle Duplicate Records
                </div>
        """, unsafe_allow_html=True)
        
        if duplicate_rows == 0:
            st.success("No duplicate records detected in the current dataset.")
        else:
            st.warning(f"Detected {duplicate_rows} duplicate record(s) in the dataset.")
            if st.button("Deduplicate Dataset", use_container_width=True):
                cleaned = clean_duplicate_records(raw_data)
                st.session_state["raw_df"] = cleaned
                st.success("Duplicate records removed successfully!")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Statistical Data Summary
    st.markdown("---")
    st.markdown("<div style='font-size: 1.15rem; font-weight: 700; color: #1E3A5F; margin-bottom: 12px;'>Comprehensive Descriptive Statistics (Numerical Features)</div>", unsafe_allow_html=True)
    stats_summary = generate_numerical_summary_table(active_df)
    st.dataframe(stats_summary, hide_index=True, use_container_width=True)


# ==============================================================================
# PAGE 6: ABOUT PROJECT PAGE
# ==============================================================================
elif nav_selection.startswith("6."):
    st.markdown("""
        <div class="section-banner">
            <h2>About Student Performance Analysis System</h2>
            <p>Academic micro-project documentation, architecture, and applied Data Science methodologies.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="content-card" style="margin-bottom: 22px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                <h3 style="color: #1E3A5F; margin: 0; font-size: 1.25rem;">1. Project Objective & Academic Context</h3>
                <span class="info-chip">GTU Sem 5 &bull; PDS Micro-Project</span>
            </div>
            <p style="line-height: 1.65; color: #374151; margin-bottom: 10px;">
                The <b>Student Performance Analysis System</b> is an end-to-end Python-based Data Science micro-project 
                developed for Gujarat Technological University (GTU) Semester 5 Python for Data Science (PDS). 
                The primary purpose is to demonstrate practical applications of exploratory data analysis, data cleaning, 
                inferential statistical modeling, and data visualization on authentic student examination records.
            </p>
            <p style="line-height: 1.65; color: #374151; margin-bottom: 0;">
                Unlike commercial Student Information Systems (SIS) focused on transactional CRUD operations, this system 
                prioritizes <b>analytical depth</b>: extracting meaningful patterns from multi-subject academic records, 
                evaluating assessment reliability, and diagnosing individual learning strengths and weaknesses.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_tech, col_concepts = st.columns(2)

    with col_tech:
        st.markdown("""
            <div class="tech-card">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
                    <h4 style="color: #1E3A5F; margin: 0; font-size: 1.1rem;">2. Technology Stack</h4>
                </div>
                <ul style="color: #374151; line-height: 1.8; margin-bottom: 0; padding-left: 20px;">
                    <li><b>Python 3.x:</b> Core programming language</li>
                    <li><b>Pandas:</b> Data ingestion, aggregation, grouping, pivot tables, and data wrangling</li>
                    <li><b>NumPy:</b> Vectorized numerical calculations and statistical arrays</li>
                    <li><b>SciPy:</b> Pearson correlation coefficients, p-values, skewness, and kurtosis calculations</li>
                    <li><b>Matplotlib &amp; Seaborn:</b> Strict flat-color academic charts and distribution plots</li>
                    <li><b>Streamlit:</b> Reactive web-based analytical user interface</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col_concepts:
        st.markdown("""
            <div class="tech-card">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
                    <h4 style="color: #1E3A5F; margin: 0; font-size: 1.1rem;">3. Core Data Science Concepts</h4>
                </div>
                <ul style="color: #374151; line-height: 1.8; margin-bottom: 0; padding-left: 20px;">
                    <li><b>Data Cleaning:</b> Identification and remediation of null values and duplicate records</li>
                    <li><b>Exploratory Data Analysis (EDA):</b> Multi-dimensional analysis across subjects and cohorts</li>
                    <li><b>Descriptive Statistics:</b> Mean, median, mode, variance, standard deviation, and quartiles</li>
                    <li><b>Correlation Analysis:</b> Quantifying linear relationships without presupposing causation</li>
                    <li><b>Rule-Based Insight Generation:</b> Diagnostic algorithms mapping raw marks to actionable academic advice</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="content-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
                <h4 style="color: #1E3A5F; margin: 0; font-size: 1.1rem;">4. System Workflow Pipeline</h4>
            </div>
            <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 6px; padding: 4px 0;">
                <div class="pipeline-step">1. Raw CSV Ingestion</div>
                <span class="pipeline-arrow">&rarr;</span>
                <div class="pipeline-step">2. Schema Normalization</div>
                <span class="pipeline-arrow">&rarr;</span>
                <div class="pipeline-step">3. Cleaning &amp; Imputation</div>
                <span class="pipeline-arrow">&rarr;</span>
                <div class="pipeline-step">4. Feature Calculation</div>
                <span class="pipeline-arrow">&rarr;</span>
                <div class="pipeline-step">5. Statistical Modeling</div>
                <span class="pipeline-arrow">&rarr;</span>
                <div class="pipeline-step">6. Visualizations</div>
                <span class="pipeline-arrow">&rarr;</span>
                <div class="pipeline-step">7. Interactive UI</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)

    # Modular Codebase Architecture
    st.markdown("""
        <div class="content-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
                <h4 style="color: #1E3A5F; margin: 0; font-size: 1.1rem;">5. Modular Project Architecture</h4>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-top: 10px;">
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px;">
                    <div style="font-weight: 700; color: #1E3A5F; font-size: 0.92rem; margin-bottom: 4px;"><code>utils/data_processing.py</code></div>
                    <div style="font-size: 0.82rem; color: #64748B;">Schema normalization, flexible mapping, grade assignment, aggregate summaries, missing value &amp; duplicate handlers.</div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px;">
                    <div style="font-weight: 700; color: #1E3A5F; font-size: 0.92rem; margin-bottom: 4px;"><code>utils/statistics.py</code></div>
                    <div style="font-size: 0.82rem; color: #64748B;">Descriptive statistics, SciPy Pearson correlation coefficients with p-values, and subject-wise score metrics.</div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px;">
                    <div style="font-weight: 700; color: #1E3A5F; font-size: 0.92rem; margin-bottom: 4px;"><code>utils/insights.py</code></div>
                    <div style="font-size: 0.82rem; color: #64748B;">Rule-based diagnostic algorithms generating programmatic observations and actionable academic guidance.</div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px;">
                    <div style="font-weight: 700; color: #1E3A5F; font-size: 0.92rem; margin-bottom: 4px;"><code>charts/visualizations.py</code></div>
                    <div style="font-size: 0.82rem; color: #64748B;">Academic Matplotlib and Seaborn figures adhering strictly to high-contrast flat styling without decorative gradients.</div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px;">
                    <div style="font-weight: 700; color: #1E3A5F; font-size: 0.92rem; margin-bottom: 4px;"><code>app.py</code></div>
                    <div style="font-size: 0.82rem; color: #64748B;">Interactive Streamlit dashboard orchestrating multi-page views, cohort filtering, caching, and state management.</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)



# ==============================================================================
# Application Footer
# ==============================================================================
st.markdown("""
    <div class="app-footer">
        Student Performance Analysis System &bull; Python for Data Science (PDS) Micro-Project &bull; Flat Design Interface
    </div>
""", unsafe_allow_html=True)
