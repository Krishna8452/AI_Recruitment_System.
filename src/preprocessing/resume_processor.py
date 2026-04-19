"""
preprocessing/resume_processor.py
Extracts structured features from resume text using NLP.
Handles both free-text resumes and structured CSV fields.
"""

import re
import string
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Skills keyword list for matching
TECHNICAL_SKILLS = {
    "python", "java", "sql", "r", "c++", "javascript", "react", "aws",
    "docker", "kubernetes", "tensorflow", "pytorch", "machine learning",
    "deep learning", "nlp", "natural language processing", "data analysis",
    "statistics", "tableau", "excel", "git", "agile", "flask", "django",
    "spark", "hadoop", "mongodb", "postgresql", "azure", "gcp"
}

SOFT_SKILLS = {
    "communication", "leadership", "teamwork", "problem solving",
    "project management", "time management", "critical thinking",
    "collaboration", "adaptability", "creativity", "analytical"
}

DEGREE_MAP = {
    "phd":       4,
    "doctorate": 4,
    "master":    3,
    "msc":       3,
    "mba":       3,
    "bachelor":  2,
    "bsc":       2,
    "ba":        2,
    "diploma":   1,
    "none":      0,
}


class ResumeProcessor:
    """
    Converts raw resume text and structured fields into
    a numeric feature matrix for ML model input.
    """

    def __init__(self):
        self.feature_names_ = None

    def extract_text_features(self, text: str) -> dict:
        """Extract features from raw resume text."""
        if not isinstance(text, str):
            text = ""

        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        # Count technical and soft skill matches
        tech_matches = TECHNICAL_SKILLS.intersection(words)
        soft_matches  = SOFT_SKILLS.intersection(words)

        # Extract years of experience from text if mentioned
        exp_pattern = re.findall(r'(\d+)\s*(?:\+\s*)?years?\s*(?:of\s*)?(?:experience|exp)', text_lower)
        text_years_exp = int(exp_pattern[0]) if exp_pattern else 0

        # Word count as proxy for resume completeness
        word_count = len(text.split())

        return {
            "text_tech_skill_count": len(tech_matches),
            "text_soft_skill_count": len(soft_matches),
            "text_years_experience": text_years_exp,
            "resume_word_count":     word_count,
        }

    def encode_degree(self, degree: str) -> int:
        """Map degree string to ordinal value."""
        if not isinstance(degree, str):
            return 0
        d = degree.lower().strip()
        for key, val in DEGREE_MAP.items():
            if key in d:
                return val
        return 0

    def extract_skill_flags(self, skills_str: str) -> dict:
        """Create binary flags for key skills from pipe-delimited skill string."""
        if not isinstance(skills_str, str):
            return {}
        skills_lower = set(s.strip().lower() for s in skills_str.split("|"))
        flags = {}
        for skill in ["Python", "Machine Learning", "SQL", "Java",
                      "Deep Learning", "Communication", "Leadership",
                      "Project Management", "Data Analysis"]:
            flags[f"skill_{skill.lower().replace(' ', '_')}"] = (
                1 if skill.lower() in skills_lower else 0
            )
        return flags

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw DataFrame into feature matrix.
        Expects columns: resume_text, degree, gpa, years_experience,
                         num_skills, skills, prev_companies
        """
        features = pd.DataFrame()

        # ── Structured features ───────────────────────────────────────────────
        features["degree_encoded"]    = df["degree"].apply(self.encode_degree)
        features["gpa"]               = df["gpa"].fillna(0).clip(0, 4)
        features["years_experience"]  = df["years_experience"].fillna(0).clip(0, 30)
        features["num_skills"]        = df["num_skills"].fillna(0)
        features["prev_companies"]    = df["prev_companies"].fillna(0)

        # ── Text-derived features ─────────────────────────────────────────────
        text_features = df["resume_text"].apply(
            lambda t: pd.Series(self.extract_text_features(t))
        )
        features = pd.concat([features, text_features], axis=1)

        # ── Skill flags ───────────────────────────────────────────────────────
        if "skills" in df.columns:
            skill_flags = df["skills"].apply(
                lambda s: pd.Series(self.extract_skill_flags(s))
            ).fillna(0)
            features = pd.concat([features, skill_flags], axis=1)

        # ── Interaction features ───────────────────────────────────────────────
        features["exp_x_degree"]     = features["years_experience"] * features["degree_encoded"]
        features["gpa_x_skills"]     = features["gpa"] * features["num_skills"]

        self.feature_names_ = list(features.columns)
        return features.astype(float)

    def get_feature_names(self) -> list:
        if self.feature_names_ is None:
            raise ValueError("Call process_dataframe first.")
        return self.feature_names_


if __name__ == "__main__":
    # Quick test
    df = pd.read_csv("data/AI_Resume_Screening.csv.csv")
    processor = ResumeProcessor()
    X = processor.process_dataframe(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {processor.get_feature_names()}")
    print(X.describe().round(3))
