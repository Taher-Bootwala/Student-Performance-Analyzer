# Student Performance Analysis System
**Python for Data Science (PDS) Micro-Project | GTU Semester 5 Submission**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-1E3A5F?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-1E3A5F?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-1E3A5F?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized%20Math-1E3A5F?style=flat&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-Statistical%20Inference-1E3A5F?style=flat&logo=scipy&logoColor=white)](https://scipy.org)
[![Matplotlib & Seaborn](https://img.shields.io/badge/Visualizations-Flat%20Design-1E3A5F?style=flat)](https://matplotlib.org)

---

## 1. Project Introduction

The **Student Performance Analysis System** is a dedicated Data Science micro-project designed to transform raw student academic records into actionable educational insights, statistical metrics, performance diagnostics, and high-clarity visualizations.

Unlike transactional Student Information Systems (SIS) focused on admin tasks (such as fee collection or attendance logging), this project focuses squarely on **Data Science methodology**:
1. **Data Ingestion & Cleaning** (handling missing values, deduplication, schema normalization).
2. **Exploratory Data Analysis (EDA)** (multi-subject performance distributions, outlier detection).
3. **Descriptive & Inferential Statistics** (measures of central tendency, dispersion, and Pearson correlation coefficients with hypothesis testing).
4. **Data Visualization** (academic flat-color charts strictly avoiding gradients or neon styling).
5. **Rule-Based Diagnostic Intelligence** (automated identification of learning strengths, subject anomalies, and remedial requirements).

---

## 2. Key Features & Page Overview

The system features an interactive sidebar navigation divided into 6 distinct analytical modules:

```text
┌─────────────────────────────────────────────────────────────┐
│ Student Performance Analyzer                                │
├─────────────────────────────────────────────────────────────┤
│ 1. Dashboard            - Cohort KPIs, Donut & Bar Charts   │
│ 2. Student Analysis     - Individual Profile & Diagnostics  │
│ 3. Subject Analysis     - Histograms, Boxplots & Watchlists │
│ 4. Performance Trends   - Scatter Plots & Correlation Heatmap│
│ 5. Dataset              - Data Inspector & In-Memory Cleaner│
│ 6. About Project        - Academic Scope & Technical Pipeline│
└─────────────────────────────────────────────────────────────┘
```

### Module 1: Overview Dashboard
- **5 High-Impact KPI Cards**:
  - `Total Students`: Active cohort count (e.g. 120 students).
  - `Average Attendance`: Cohort average attendance percentage.
  - `Average Marks`: Aggregate mean score across all subjects.
  - `Pass Percentage`: Ratio of students passing all curricular papers.
  - `Top Performer`: Highest-scoring student name, percentage, and ID.
- **Visualizations**:
  - **Pass vs Fail Donut Chart**: Proportional breakdown with counts and percentages.
  - **Average Marks by Subject**: Horizontal bar chart comparing course difficulty.
  - **Attendance vs Overall Marks**: Scatter plot with linear regression trendline.
  - **Grade Distribution**: Frequency histogram across standard grades (`A+` to `F`).
  - **Top 10 Performers**: Horizontal ranking chart sorted by overall percentage.
  - **Cohort Strategic Observations**: Programmatic batch-level insights on syllabus difficulty and attendance health.

### Module 2: Individual Student Analysis
- Searchable selection by **Student ID** or **Student Name**.
- **Student Profile Card**: Displays Name, Roll ID, Stream, Semester, Gender, Result badge (`PASS` / `FAIL`), and Classification badge (`Excellent`, `Good`, `Average`, `Needs Improvement`).
- **Subject Assessment Breakdown Table**:
  - Attendance (%), Internal Examination (out of 20), Assignment (out of 10), Practical Examination (out of 20), Final Examination (out of 50), and Total Marks (out of 100).
- **Component Stacked Bar Chart**: Visual breakdown of internal vs assignment vs practical vs final exams per subject.
- **Rule-Based Diagnostic Insights**: Automated evaluation flagging:
  - Low attendance alert (< 75%).
  - Practical vs Theory divergence (e.g., student demonstrates stronger practical aptitude than theory).
  - Subject-specific weaknesses needing tutoring.
  - Standout subject aptitudes.

### Module 3: Subject Analysis
- Subject dropdown selector (`Mathematics`, `Python Programming`, `Database Management`, etc.).
- **Summary Metrics**: Average Marks, Highest Marks, Lowest Marks, Pass Percentage, Enrolled Students, Standard Deviation.
- **Student Performance Distribution**: Histogram with continuous Kernel Density Estimation (KDE) and annotated Mean/Median lines.
- **Marks Comparison Boxplot**: Visualizing Median, Quartiles ($Q_1, Q_3$), and Outliers.
- **Top 5 Performers**: Best scoring students in the chosen subject.
- **Weak Performers Watchlist**: Interactive threshold slider to immediately identify students scoring below a given benchmark (e.g. < 40 marks).

### Module 4: Performance Trends & Correlation
- **Attendance vs Overall Marks**: Scatter plot with regression line and Pearson $r$.
- **Internal Exam vs Final Exam**: Quantifying how continuous assessments predict final summative performance.
- **Assignment vs Final Exam**: Measuring correlation between continuous coursework and end-of-term exams.
- **Correlation Matrix Heatmap**: Full 6-feature Seaborn correlation heatmap.
- **Academic Interpretation Note**: Explicitly framing relationships without presuming causation ($r$ and $p$-value explained).

### Module 5: Dataset Management & Data Cleaning
- **Metadata Summary**: Total rows, total columns, missing value count, and duplicate rows.
- **Interactive Data Table**: Searchable and sortable records.
- **In-Memory Data Cleaning Controls**:
  - Null value detection with options to:
    - *Drop rows containing null values*
    - *Impute numeric missing values with Mean*
    - *Impute numeric missing values with Median*
  - Duplicate record detection and 1-click deduplication.
  - Demo uncleaned CSV loader (`students_raw_uncleaned.csv`) for live examination demonstrations.
- **Descriptive Statistics Table**: Comprehensive summary covering Count, Mean, Median, Std Dev, Variance, Min, $Q_1$, $Q_3$, Max, and Skewness.

### Module 6: About Project
- Academic project overview, technological foundations, applied Data Science concepts, and pipeline architecture.

---

## 3. Technology Stack

| Domain | Technology / Library | Purpose |
|---|---|---|
| **Language** | Python 3.11+ | Core execution and data pipeline |
| **Data Manipulation** | Pandas | Tabular data ingestion, filtering, grouping, aggregation, reshaping |
| **Numerical Math** | NumPy | Vectorized arrays, standard deviations, clipping |
| **Statistical Inference**| SciPy (`scipy.stats`) | Pearson correlation ($r$), $p$-values, skewness, kurtosis, mode |
| **Visualization** | Matplotlib & Seaborn | Academic flat-color plots, histograms, boxplots, heatmaps |
| **Web Interface** | Streamlit | Reactive micro-project web application dashboard |
| **Data Storage** | CSV (Flat File) | Primary tabular data format |

---

## 4. UI/UX Design System (Strict Academic Flat Design)

Adhering strictly to project specifications, the interface implements **zero gradients, zero neon colors, and zero glassmorphism**:

```text
Background:   #F8FAFC  (Crisp Soft Light Slate)
Primary:      #1E3A5F  (Authoritative Academic Navy)
Text Dark:    #1F2937  (Charcoal)
Secondary:    #64748B  (Muted Slate Gray)
Borders:      #E2E8F0  (Subtle Gray Dividers)
Success:      #16A34A  (Academic Green for Pass & Distinction)
Warning:      #D97706  (Muted Amber for Remedial Alerts)
Danger:       #DC2626  (Crimson Red for Fail & Deficits)
```

---

## 5. Dataset Architecture

The default dataset (`data/students.csv`) models multi-subject semester records (120 students across 5 subjects = 600 records):

```csv
student_id,student_name,gender,department,semester,subject,attendance,internal_marks,assignment_marks,practical_marks,final_marks
ST1001,Rachit Panchal,Female,Information Technology,Semester 5,Mathematics,74.5,11,4,9,23
ST1001,Rachit Panchal,Female,Information Technology,Semester 5,Python Programming,79.3,12,6,14,34
...
```

### Assessment Weightage per Subject:
- **Internal Assessment**: Max 20 marks
- **Assignment / Coursework**: Max 10 marks
- **Practical Laboratory**: Max 20 marks
- **Final University Exam**: Max 50 marks
- **Subject Total**: $\mathbf{100\text{ marks}}$

### Universal CSV Adaptor:
The loader automatically maps standard column variations and includes compatibility with the Kaggle/UCI student performance dataset (`G1`, `G2`, `G3`, `absences`).

---

## 6. Configurable Grading & Classification System

The grading logic is centralized in [`utils/data_processing.py`](file:///d:/GTU/Degree/Sem_5/Submission/PDS/MP/Student%20Performance%20Analysis/utils/data_processing.py):

| Percentage Range | Grade Awarded | Result Status |
|---|---|---|
| $90.0\% - 100\%$ | **A+** | PASS |
| $80.0\% - 89.9\%$ | **A** | PASS |
| $70.0\% - 79.9\%$ | **B+** | PASS |
| $60.0\% - 69.9\%$ | **B** | PASS |
| $50.0\% - 59.9\%$ | **C** | PASS |
| $40.0\% - 49.9\%$ | **D** | PASS |
| $< 40.0\%$ | **F** | FAIL |

### Academic Performance Categories:
- **Excellent**: Overall Percentage $\ge 80.0\%$
- **Good**: Overall Percentage $65.0\% - 79.9\%$
- **Average**: Overall Percentage $50.0\% - 64.9\%$
- **Needs Improvement**: Overall Percentage $< 50.0\%$

---

## 7. Project Directory Structure

```text
Student Performance Analysis/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Micro-project report & documentation
│
├── data/
│   ├── students.csv            # Primary 600-record academic dataset
│   ├── students_raw_uncleaned.csv # Dataset with intentional nulls/duplicates
│   └── generate_dataset.py     # Reproducible dataset generator script
│
├── utils/
│   ├── __init__.py
│   ├── data_processing.py      # Schema normalization, grading, cleaning
│   ├── statistics.py           # Descriptive stats, SciPy Pearson r, p-values
│   └── insights.py             # Rule-based diagnostic academic insight engine
│
├── charts/
│   ├── __init__.py
│   └── visualizations.py       # Matplotlib & Seaborn flat-color visualizations
│
└── assets/
    └── style.css               # Flat-design CSS styling (no gradients)
```

---

## 8. Installation & Setup Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.

### Step 1: Clone or Navigate to Directory
```bash
cd "d:\GTU\Degree\Sem_5\Submission\PDS\MP\Student Performance Analysis"
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Dashboard
```bash
streamlit run app.py
```

The application will automatically open in your default browser at:
`http://localhost:8501`

---

## 9. Data Science Concepts Demonstrated

1. **Data Preprocessing & Wrangling**:
   - Ingesting multi-subject records, resolving column synonyms, standardizing data types, handling missing values via imputation (mean/median) and removing duplicate entries.
2. **Descriptive Statistics**:
   - Central tendency (Mean, Median, Mode) and dispersion (Variance, Standard Deviation, Range, Interquartile Range $Q_3 - Q_1$, Skewness).
3. **Inferential & Correlation Analysis**:
   - Quantifying relationships between continuous testing and summative examinations via Pearson Correlation Coefficient ($r$) and significance testing ($p$-value).
4. **Data Visualization**:
   - Visualizing distribution densities with KDE, detecting outliers with Tukey boxplots, and presenting multi-component distributions using stacked bar charts.
5. **Heuristic Rule-Based Insights**:
   - Programmatically translating continuous statistical data into actionable diagnostic feedback for educators and students.

---

## 10. Future Enhancements

- Predictive Machine Learning models (e.g., Logistic Regression or Random Forest) to predict at-risk students before final examinations.
- Automated PDF report generation summarizing student grade cards for parent-teacher conferences.
- Time-series tracking across multiple academic semesters to observe longitudinal growth.
