"""
data_generator.py
Generates a synthetic recruitment dataset for experimentation.
Mirrors the structure of the Kaggle AI-powered resume screening dataset.
Includes demographic fields for fairness evaluation.

Run: python src/data_generator.py
Output: data/AI_Resume_Screening.csv.csv
"""

import random
import pandas as pd
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
N_CANDIDATES = 1000
OUTPUT_PATH   = Path("data/AI_Resume_Screening.csv.csv")

# ─── DATA POOLS ───────────────────────────────────────────────────────────────
SKILLS_POOL = [
    "Python", "Java", "SQL", "Machine Learning", "Data Analysis",
    "Communication", "Project Management", "Leadership", "TensorFlow",
    "Excel", "R", "Tableau", "AWS", "Docker", "Agile", "React",
    "JavaScript", "C++", "Statistics", "NLP", "Deep Learning",
    "Problem Solving", "Team Collaboration", "Git", "Kubernetes"
]

DEGREES = ["Bachelor", "Master", "PhD", "Diploma", "None"]
DEGREE_WEIGHTS = [0.45, 0.30, 0.10, 0.10, 0.05]

FIELDS = [
    "Computer Science", "Data Science", "Business Administration",
    "Engineering", "Mathematics", "Psychology", "Economics", "Other"
]

UNIVERSITIES = [
    "University of London", "Manchester University", "University of Leeds",
    "Birmingham University", "Sheffield University", "Bristol University",
    "Cardiff University", "Nottingham University", "Other Institution"
]

GENDERS = ["Male", "Female", "Non-binary"]
GENDER_WEIGHTS = [0.48, 0.47, 0.05]

ETHNICITIES = ["White", "Asian", "Black", "Hispanic", "Mixed", "Other"]
ETHNICITY_WEIGHTS = [0.55, 0.18, 0.13, 0.07, 0.04, 0.03]

JOB_TITLES_APPLIED = [
    "Data Scientist", "Software Engineer", "ML Engineer",
    "Data Analyst", "Product Manager", "Business Analyst"
]


def generate_resume_text(skills, years_exp, degree, field):
    """Generate a simple resume text string for NLP processing."""
    skill_str = ", ".join(skills)
    return (
        f"Experienced professional with {years_exp} years of experience. "
        f"Holds a {degree} in {field}. "
        f"Key skills include: {skill_str}. "
        f"Proven track record of delivering results in dynamic environments."
    )


def compute_label(row):
    """
    Rule-based label for ground truth shortlisting.
    Intentionally introduces slight bias on degree for realism.
    Returns 1 = shortlisted, 0 = not shortlisted.
    """
    score = 0

    # Experience
    score += min(row["years_experience"], 10) * 0.5

    # Degree
    degree_scores = {"PhD": 5, "Master": 4, "Bachelor": 3, "Diploma": 2, "None": 0}
    score += degree_scores.get(row["degree"], 0)

    # Number of relevant skills
    score += row["num_skills"] * 0.3

    # GPA contribution
    score += row["gpa"] * 1.5

    # Add small noise
    score += np.random.normal(0, 1.0)

    return 1 if score > 10.5 else 0


def generate_dataset(n: int) -> pd.DataFrame:
    records = []

    for i in range(n):
        # Demographics
        gender    = random.choices(GENDERS, weights=GENDER_WEIGHTS)[0]
        ethnicity = random.choices(ETHNICITIES, weights=ETHNICITY_WEIGHTS)[0]
        age       = random.randint(22, 55)

        # Education
        degree    = random.choices(DEGREES, weights=DEGREE_WEIGHTS)[0]
        field     = random.choice(FIELDS)
        university= random.choice(UNIVERSITIES)
        gpa       = round(random.uniform(2.0, 4.0), 2)

        # Experience
        min_exp = 0 if degree in ["Bachelor", "Diploma", "None"] else 0
        years_exp = random.randint(0, 15)

        # Skills
        n_skills = random.randint(3, 12)
        skills   = random.sample(SKILLS_POOL, n_skills)

        # Job applied for
        job_title = random.choice(JOB_TITLES_APPLIED)

        # Previous companies
        prev_companies = random.randint(0, 5)

        # Resume text
        resume_text = generate_resume_text(skills, years_exp, degree, field)

        record = {
            "candidate_id":     f"CAND-{i+1:04d}",
            "gender":           gender,
            "ethnicity":        ethnicity,
            "age":              age,
            "degree":           degree,
            "field_of_study":   field,
            "university":       university,
            "gpa":              gpa,
            "years_experience": years_exp,
            "num_skills":       n_skills,
            "skills":           "|".join(skills),
            "job_title_applied":job_title,
            "prev_companies":   prev_companies,
            "resume_text":      resume_text,
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Apply label
    df["shortlisted"] = df.apply(compute_label, axis=1)

    print(f"Generated {len(df)} candidates")
    print(f"Shortlisted: {df['shortlisted'].sum()} ({df['shortlisted'].mean()*100:.1f}%)")
    print(f"Gender distribution:\n{df['gender'].value_counts()}")
    print(f"Shortlist rate by gender:\n{df.groupby('gender')['shortlisted'].mean().round(3)}")

    return df


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(N_CANDIDATES)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset saved to {OUTPUT_PATH}")
