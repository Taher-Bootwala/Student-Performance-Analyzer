"""
Visualizations Module for Student Performance Analysis System.
Built using Matplotlib and Seaborn with strict adherence to flat-color,
academic design standards (NO GRADIENTS).
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# Strict Flat Design Color Palette
# ==========================================
COLOR_PRIMARY = "#1E3A5F"      # Dark Navy
COLOR_SECONDARY = "#64748B"    # Slate Gray
COLOR_SUCCESS = "#16A34A"      # Academic Emerald Green
COLOR_WARNING = "#D97706"      # Muted Amber
COLOR_DANGER = "#DC2626"       # Academic Crimson Red
COLOR_INFO = "#0284C7"         # Clean Sky Blue
COLOR_BG_CARD = "#FFFFFF"      # Crisp White
COLOR_BG_PAGE = "#F8FAFC"      # Soft Gray
COLOR_BORDER = "#E2E8F0"       # Border divider
COLOR_TEXT = "#1F2937"         # Dark Charcoal

GRADE_COLORS = {
    "A+": "#16A34A",
    "A": "#059669",
    "B+": "#0284C7",
    "B": "#2563EB",
    "C": "#D97706",
    "D": "#EA580C",
    "F": "#DC2626"
}


def apply_flat_theme():
    """Applies clean, flat, academic formatting to Matplotlib/Seaborn."""
    plt.rcParams.update({
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "figure.facecolor": COLOR_BG_CARD,
        "axes.facecolor": COLOR_BG_CARD,
        "axes.edgecolor": COLOR_BORDER,
        "axes.labelcolor": COLOR_TEXT,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": COLOR_PRIMARY,
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "xtick.color": COLOR_SECONDARY,
        "ytick.color": COLOR_SECONDARY,
        "grid.color": COLOR_BORDER,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "figure.autolayout": True
    })


# -----------------------------------------------------------------------------
# Chart 1: Pass vs Fail (Donut Chart)
# -----------------------------------------------------------------------------
def plot_pass_fail_donut(summary_df: pd.DataFrame) -> plt.Figure:
    """
    Renders a clean, flat-color donut chart showing Pass vs Fail proportion.
    """
    apply_flat_theme()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    counts = summary_df["overall_result"].value_counts()
    pass_count = counts.get("PASS", 0)
    fail_count = counts.get("FAIL", 0)
    total = pass_count + fail_count

    labels = ["Passed", "Failed"]
    sizes = [pass_count, fail_count]
    colors = [COLOR_SUCCESS, COLOR_DANGER]

    # Donut pie
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * total / 100))})",
        pctdistance=0.72,
        colors=colors,
        startangle=140,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2.5)
    )

    for t in texts:
        t.set_fontsize(11)
        t.set_color(COLOR_TEXT)
        t.set_weight("bold")
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_weight("bold")

    # Center circle for clean donut aesthetic
    centre_circle = plt.Circle((0, 0), 0.52, fc=COLOR_BG_CARD)
    ax.add_artist(centre_circle)
    ax.text(0, 0, f"Total\n{total}", ha="center", va="center", fontsize=12, fontweight="bold", color=COLOR_PRIMARY)

    ax.set_title("Pass vs Fail Proportion", pad=15)
    ax.axis("equal")
    return fig


# -----------------------------------------------------------------------------
# Chart 2: Average Marks by Subject (Horizontal Bar Chart)
# -----------------------------------------------------------------------------
def plot_subject_averages(df: pd.DataFrame) -> plt.Figure:
    """
    Renders a flat horizontal bar chart showing average marks per subject.
    """
    apply_flat_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    sub_means = df.groupby("subject")["total_marks"].mean().sort_values(ascending=True)

    y_pos = np.arange(len(sub_means))
    bars = ax.barh(y_pos, sub_means.values, color=COLOR_PRIMARY, height=0.55, edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub_means.index, fontsize=10, fontweight="medium")
    ax.set_xlabel("Average Marks (out of 100)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate exact values on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color=COLOR_PRIMARY
        )

    ax.set_title("Average Marks by Subject", pad=12)
    return fig


# -----------------------------------------------------------------------------
# Chart 3: Attendance vs Overall Marks (Scatter Plot + Trendline)
# -----------------------------------------------------------------------------
def plot_attendance_vs_marks(summary_df: pd.DataFrame) -> plt.Figure:
    """
    Renders a scatter plot showing relationship between Attendance and Overall Marks.
    """
    apply_flat_theme()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = summary_df["attendance"]
    y = summary_df["overall_marks"]

    # Scatter dots with flat color
    ax.scatter(x, y, color=COLOR_PRIMARY, alpha=0.65, edgecolors=COLOR_BORDER, s=55, zorder=3, label="Student")

    # Fit a simple linear regression trend line
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_trend, p(x_trend), color=COLOR_DANGER, linestyle="-", linewidth=2.2, zorder=4, label="Trendline")

    ax.set_xlabel("Attendance Percentage (%)", fontsize=11)
    ax.set_ylabel("Overall Marks (out of 100)", fontsize=11)
    ax.set_xlim(min(45, x.min() - 2), 102)
    ax.set_ylim(min(20, y.min() - 5), 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor=COLOR_BG_CARD, edgecolor=COLOR_BORDER, fontsize=9)
    ax.set_title("Attendance % vs Overall Academic Marks", pad=12)

    return fig


# -----------------------------------------------------------------------------
# Chart 4: Grade Distribution (Bar Chart)
# -----------------------------------------------------------------------------
def plot_grade_distribution(summary_df: pd.DataFrame) -> plt.Figure:
    """
    Renders a clean bar chart showing frequency of each academic grade.
    """
    apply_flat_theme()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    grade_order = ["A+", "A", "B+", "B", "C", "D", "F"]
    grade_counts = summary_df["grade"].value_counts()
    counts = [grade_counts.get(g, 0) for g in grade_order]
    bar_colors = [GRADE_COLORS.get(g, COLOR_PRIMARY) for g in grade_order]

    x_pos = np.arange(len(grade_order))
    bars = ax.bar(x_pos, counts, color=bar_colors, width=0.58, edgecolor="none")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(grade_order, fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Students", fontsize=11)
    ax.set_ylim(0, max(counts, default=10) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max(counts, default=10) * 0.03),
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=COLOR_TEXT
        )

    ax.set_title("Cohort Grade Distribution", pad=12)
    return fig


# -----------------------------------------------------------------------------
# Chart 5: Top 10 Students (Horizontal Bar Chart)
# -----------------------------------------------------------------------------
def plot_top_students(summary_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """
    Renders a horizontal bar chart displaying top performers.
    """
    apply_flat_theme()
    fig, ax = plt.subplots(figsize=(8, 5))

    top_df = summary_df.nsmallest(top_n, "rank").sort_values(by="overall_percentage", ascending=True)

    labels = [f"{row['student_name']} ({row['student_id']})" for _, row in top_df.iterrows()]
    y_pos = np.arange(len(top_df))

    bars = ax.barh(y_pos, top_df["overall_percentage"], color=COLOR_PRIMARY, height=0.55)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Overall Percentage (%)", fontsize=11)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color=COLOR_PRIMARY
        )

    ax.set_title(f"Top {len(top_df)} Students by Academic Percentage", pad=12)
    return fig


# -----------------------------------------------------------------------------
# Chart 6: Student Subject Breakdown (Component Stacked Bar Chart)
# -----------------------------------------------------------------------------
def plot_student_subject_breakdown(student_subjects_df: pd.DataFrame, class_df: Optional[pd.DataFrame] = None) -> plt.Figure:
    """
    Renders a stacked bar chart displaying marks components (Internal, Assignment, Practical, Final)
    for an individual student across all subjects.
    """
    apply_flat_theme()
    fig, ax = plt.subplots(figsize=(9, 4.8))

    subjects = student_subjects_df["subject"].tolist()
    y_pos = np.arange(len(subjects))
    
    internals = student_subjects_df["internal_marks"].values
    assignments = student_subjects_df["assignment_marks"].values
    practicals = student_subjects_df["practical_marks"].values
    finals = student_subjects_df["final_marks"].values

    # Flat, distinct segment colors
    c_internal = "#1E3A5F"   # Navy (Max 20)
    c_assignment = "#0284C7" # Sky Blue (Max 10)
    c_practical = "#0D9488"  # Teal (Max 20)
    c_final = "#475569"      # Slate (Max 50)

    height = 0.52

    b1 = ax.barh(y_pos, internals, height=height, label="Internal (20)", color=c_internal)
    b2 = ax.barh(y_pos, assignments, left=internals, height=height, label="Assignment (10)", color=c_assignment)
    b3 = ax.barh(y_pos, practicals, left=internals + assignments, height=height, label="Practical (20)", color=c_practical)
    b4 = ax.barh(y_pos, finals, left=internals + assignments + practicals, height=height, label="Final Exam (50)", color=c_final)

    totals = internals + assignments + practicals + finals
    for idx, total in enumerate(totals):
        ax.text(total + 1.5, y_pos[idx], f"{int(total)}/100", va="center", ha="left", fontsize=9.5, fontweight="bold", color=COLOR_PRIMARY)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(subjects, fontsize=10, fontweight="medium")
    ax.set_xlabel("Marks Scored (out of 100)", fontsize=11)
    ax.set_xlim(0, 112)
    ax.axvline(40, color=COLOR_DANGER, linestyle=":", linewidth=1.5, label="Pass Mark (40)")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor=COLOR_BG_CARD, edgecolor=COLOR_BORDER, fontsize=8.5)
    ax.set_title("Subject-wise Marks Component Breakdown", pad=12)

    return fig


# -----------------------------------------------------------------------------
# Chart 7: Subject Performance Distribution (Histogram with KDE)
# -----------------------------------------------------------------------------
def plot_subject_distribution(df: pd.DataFrame, subject: str) -> plt.Figure:
    """
    Renders histogram and KDE distribution curve for marks in a single subject.
    """
    apply_flat_theme()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    sub_data = df[df["subject"] == subject]["total_marks"].dropna()

    sns.histplot(
        sub_data,
        bins=10,
        kde=True,
        ax=ax,
        color=COLOR_PRIMARY,
        edgecolor=COLOR_BG_CARD,
        linewidth=1.2,
        line_kws={"color": COLOR_DANGER, "linewidth": 2}
    )

    mean_val = sub_data.mean()
    median_val = sub_data.median()

    ax.axvline(mean_val, color=COLOR_WARNING, linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color=COLOR_SUCCESS, linestyle="-.", linewidth=1.8, label=f"Median: {median_val:.1f}")
    ax.axvline(40, color=COLOR_DANGER, linestyle=":", linewidth=1.5, label="Pass Mark (40)")

    ax.set_xlabel(f"{subject} Total Marks", fontsize=11)
    ax.set_ylabel("Student Frequency", fontsize=11)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor=COLOR_BG_CARD, edgecolor=COLOR_BORDER, fontsize=8.5)
    ax.set_title(f"Score Distribution: {subject}", pad=12)

    return fig


# -----------------------------------------------------------------------------
# Chart 8: Subject Marks Comparison (Box Plot)
# -----------------------------------------------------------------------------
def plot_subject_boxplot(df: pd.DataFrame, subject: str) -> plt.Figure:
    """
    Renders a box plot showing Median, Quartiles, and Outliers for a subject.
    """
    apply_flat_theme()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    sub_data = df[df["subject"] == subject]["total_marks"].dropna()

    box = ax.boxplot(
        sub_data,
        vert=False,
        patch_artist=True,
        widths=0.45,
        boxprops=dict(facecolor="#E2E8F0", color=COLOR_PRIMARY, linewidth=1.8),
        medianprops=dict(color=COLOR_DANGER, linewidth=2.2),
        whiskerprops=dict(color=COLOR_PRIMARY, linewidth=1.5),
        capprops=dict(color=COLOR_PRIMARY, linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor=COLOR_WARNING, markersize=7, markeredgecolor=COLOR_TEXT)
    )

    q1 = sub_data.quantile(0.25)
    med = sub_data.median()
    q3 = sub_data.quantile(0.75)

    ax.set_yticks([])
    ax.set_xlabel("Total Marks (out of 100)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Annotated markers
    ax.text(q1, 1.28, f"Q1: {q1:.1f}", ha="center", fontsize=9, fontweight="bold", color=COLOR_PRIMARY)
    ax.text(med, 0.72, f"Median: {med:.1f}", ha="center", fontsize=9, fontweight="bold", color=COLOR_DANGER)
    ax.text(q3, 1.28, f"Q3: {q3:.1f}", ha="center", fontsize=9, fontweight="bold", color=COLOR_PRIMARY)

    ax.set_title(f"Quartiles & Outlier Analysis: {subject}", pad=12)
    return fig


# -----------------------------------------------------------------------------
# Chart 9: Correlation Matrix Heatmap
# -----------------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Renders a clean Seaborn heatmap of the correlation matrix between numerical features.
    """
    apply_flat_theme()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    target_cols = ["attendance", "internal_marks", "assignment_marks", "practical_marks", "final_marks", "total_marks"]
    valid_cols = [c for c in target_cols if c in df.columns]
    
    corr = df[valid_cols].corr()

    # Clean label mapping for display
    display_names = {
        "attendance": "Attendance",
        "internal_marks": "Internal",
        "assignment_marks": "Assignment",
        "practical_marks": "Practical",
        "final_marks": "Final Exam",
        "total_marks": "Total Marks"
    }
    corr_display = corr.rename(index=display_names, columns=display_names)

    sns.heatmap(
        corr_display,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=1.2,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 10, "weight": "bold", "color": COLOR_TEXT}
    )

    ax.set_title("Academic Features Correlation Heatmap", pad=12)
    return fig


# -----------------------------------------------------------------------------
# Chart 10: Generic Scatter Plot with Regression Trendline
# -----------------------------------------------------------------------------
def plot_scatter_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    r_val: Optional[float] = None
) -> plt.Figure:
    """
    Renders an analytical scatter plot with a linear regression trend line and correlation tag.
    """
    apply_flat_theme()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    valid = df[[x_col, y_col]].dropna()
    x = valid[x_col]
    y = valid[y_col]

    ax.scatter(x, y, color=COLOR_PRIMARY, alpha=0.6, edgecolors=COLOR_BORDER, s=50, label="Observations")

    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, color=COLOR_DANGER, linewidth=2, label="Trendline")

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    if r_val is not None:
        ax.text(
            0.05, 0.90,
            f"Pearson r = {r_val:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=COLOR_BG_CARD, edgecolor=COLOR_BORDER)
        )

    ax.legend(loc="lower right", frameon=True, facecolor=COLOR_BG_CARD, edgecolor=COLOR_BORDER, fontsize=8.5)
    ax.set_title(title, pad=12)

    return fig
