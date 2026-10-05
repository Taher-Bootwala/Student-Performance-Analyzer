"""
Dataset Generator for Student Performance Analysis System.
Generates realistic multi-subject academic records for 120 students.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

first_names = [
    "Rahul", "Priya", "Aarav", "Aisha", "Rohan", "Ananya", "Devansh", "Sneha",
    "Kabir", "Tanvi", "Aditya", "Neha", "Karan", "Diya", "Manish", "Ishaan",
    "Pooja", "Vikram", "Riya", "Yash", "Meera", "Arjun", "Simran", "Siddharth",
    "Kavya", "Harsh", "Shruti", "Varun", "Bhavya", "Akash", "Anushka", "Nikhil",
    "Payal", "Pranav", "Disha", "Gaurav", "Sanya", "Kunal", "Rachit", "Natasha"
]

last_names = [
    "Patel", "Sharma", "Mehta", "Khan", "Desai", "Shah", "Joshi", "Verma",
    "Nair", "Dave", "Gupta", "Choudhury", "Bhatt", "Trivedi", "Iyer", "Reddy",
    "Malhotra", "Kapoor", "Vyas", "Panchal", "Mistry", "Pandya", "Kulkarni", "Chauhan"
]

subjects = [
    "Mathematics",
    "Python Programming",
    "Database Management",
    "Computer Networks",
    "Operating Systems"
]

departments = ["Computer Engineering", "Information Technology", "Data Science"]
genders = ["Male", "Female"]

num_students = 120
students = []

# Generate unique students
for i in range(1, num_students + 1):
    sid = f"ST{1000 + i}"
    fn = np.random.choice(first_names)
    ln = np.random.choice(last_names)
    name = f"{fn} {ln}"
    gender = np.random.choice(genders, p=[0.55, 0.45])
    dept = np.random.choice(departments, p=[0.40, 0.35, 0.25])
    
    # Base capability / study habit factor (latent variable)
    base_ability = np.random.beta(5, 2.5)  # between 0 and 1, skewed toward 0.65-0.85
    
    # Base attendance tied moderately to ability with noise
    base_att = np.clip(55 + (base_ability * 40) + np.random.normal(0, 5), 52, 98)
    
    students.append({
        "student_id": sid,
        "student_name": name,
        "gender": gender,
        "department": dept,
        "semester": "Semester 5",
        "base_ability": base_ability,
        "base_attendance": round(float(base_att), 1)
    })

records = []
for s in students:
    # Student attendance varies slightly across subjects
    for sub in subjects:
        # Subject difficulty factor
        diff_bias = {
            "Mathematics": -4,
            "Python Programming": +3,
            "Database Management": +1,
            "Computer Networks": -1,
            "Operating Systems": 0
        }[sub]
        
        att = np.clip(s["base_attendance"] + np.random.normal(0, 2.5), 50, 100)
        att = round(float(att), 1)
        
        # Performance derived from ability, attendance and noise
        perf_score = s["base_ability"] * 100 + (att - 75) * 0.2 + diff_bias + np.random.normal(0, 4)
        perf_pct = np.clip(perf_score, 25, 98) / 100.0
        
        # Internal marks out of 20
        internal = np.clip(np.round(perf_pct * 20 + np.random.normal(0, 1.2)), 5, 20)
        # Assignment marks out of 10
        assignment = np.clip(np.round(perf_pct * 10 + np.random.normal(0, 0.8)), 3, 10)
        # Practical marks out of 20
        practical = np.clip(np.round(perf_pct * 20 + np.random.normal(0, 1.2)), 6, 20)
        # Final marks out of 50
        final = np.clip(np.round(perf_pct * 50 + np.random.normal(0, 3.0)), 12, 50)
        
        records.append({
            "student_id": s["student_id"],
            "student_name": s["student_name"],
            "gender": s["gender"],
            "department": s["department"],
            "semester": s["semester"],
            "subject": sub,
            "attendance": att,
            "internal_marks": int(internal),
            "assignment_marks": int(assignment),
            "practical_marks": int(practical),
            "final_marks": int(final)
        })

df = pd.DataFrame(records)
df.to_csv("data/students.csv", index=False)
print(f"Generated clean dataset: {len(df)} records ({len(students)} students).")

# Also generate a dataset with some duplicates and missing values for demonstrating cleaning
df_dirty = df.copy()
# Add 4 duplicate rows
dups = df_dirty.iloc[[10, 25, 40, 70]]
df_dirty = pd.concat([df_dirty, dups], ignore_index=True)

# Add some missing values
missing_indices = [5, 18, 33, 52, 88, 110]
for idx in missing_indices:
    col = np.random.choice(["attendance", "internal_marks", "practical_marks", "final_marks"])
    df_dirty.loc[idx, col] = np.nan

df_dirty.to_csv("data/students_raw_uncleaned.csv", index=False)
print(f"Generated uncleaned dataset with duplicates and nulls: {len(df_dirty)} records.")
