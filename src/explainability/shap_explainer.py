"""
explainability/shap_explainer.py
Uses SHAP (SHapley Additive exPlanations) to explain model decisions.

Produces:
  - Global feature importance (summary plot)
  - Individual candidate explanation (waterfall / force plot)
  - SHAP values CSV for all test candidates
  - Text explanations for recruiters

Reference: Lundberg & Lee (2017)
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocessing.resume_processor import ResumeProcessor

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Feature display name mapping for human-readable explanations
FEATURE_DISPLAY_NAMES = {
    "degree_encoded":             "Education Level",
    "gpa":                        "GPA Score",
    "years_experience":           "Years of Experience",
    "num_skills":                 "Number of Skills",
    "prev_companies":             "Previous Employers",
    "text_tech_skill_count":      "Technical Skills (Text)",
    "text_soft_skill_count":      "Soft Skills (Text)",
    "text_years_experience":      "Experience (Mentioned in CV)",
    "resume_word_count":          "CV Completeness (Word Count)",
    "exp_x_degree":               "Experience × Education",
    "gpa_x_skills":               "GPA × Skills",
    "skill_python":               "Python Skill",
    "skill_machine_learning":     "Machine Learning Skill",
    "skill_sql":                  "SQL Skill",
    "skill_java":                 "Java Skill",
    "skill_deep_learning":        "Deep Learning Skill",
    "skill_communication":        "Communication Skill",
    "skill_leadership":           "Leadership Skill",
    "skill_project_management":   "Project Management",
    "skill_data_analysis":        "Data Analysis Skill",
}


class SHAPExplainer:
    """
    Wraps SHAP TreeExplainer/LinearExplainer to explain
    recruitment model decisions at both global and local level.
    """

    def __init__(self):
        self.model     = joblib.load(MODELS_DIR / "best_model.pkl")
        self.scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
        self.processor = ResumeProcessor()
        self.explainer = None
        self.shap_values = None
        self.X_sc_       = None
        self.feature_names_ = None

        with open(MODELS_DIR / "model_info.json") as f:
            info = json.load(f)
        self.model_name = info.get("best_model_name", "Model")

    def fit(self, csv_path: str = "data/AI_Resume_Screening.csv.csv"):
        """
        Compute SHAP values for the full dataset.
        Uses TreeExplainer for tree models, LinearExplainer for logistic regression.
        """
        import shap

        df = pd.read_csv(csv_path)
        X  = self.processor.process_dataframe(df)
        self.feature_names_ = self.processor.get_feature_names()
        self.X_sc_ = self.scaler.transform(X)
        self.df_   = df

        print(f"Computing SHAP values for {self.model_name}...")

        model_type = type(self.model).__name__
        if model_type in ["RandomForestClassifier", "GradientBoostingClassifier",
                          "DecisionTreeClassifier"]:
            self.explainer = shap.TreeExplainer(self.model)
            sv = self.explainer.shap_values(self.X_sc_)
            # Tree models return list [class0, class1]; take class 1
            self.shap_values = sv[1] if isinstance(sv, list) else sv
        else:
            # Logistic Regression
            self.explainer = shap.LinearExplainer(
                self.model, self.X_sc_, feature_perturbation="correlation_dependent"
            )
            self.shap_values = self.explainer.shap_values(self.X_sc_)

        print(f"SHAP values computed. Shape: {self.shap_values.shape}")
        self._save_shap_csv()
        return self

    def _save_shap_csv(self):
        """Save mean absolute SHAP values per feature."""
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        df_shap = pd.DataFrame({
            "feature":    self.feature_names_,
            "display_name": [FEATURE_DISPLAY_NAMES.get(f, f) for f in self.feature_names_],
            "mean_abs_shap": mean_abs.round(5),
        }).sort_values("mean_abs_shap", ascending=False)
        df_shap.to_csv(RESULTS_DIR / "shap_feature_importance.csv", index=False)
        print("Saved: results/shap_feature_importance.csv")

    def plot_global_summary(self):
        """
        SHAP summary plot showing feature importance and direction
        of effect across all candidates.
        """
        import shap
        display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in self.feature_names_]

        plt.figure(figsize=(9, 7))
        shap.summary_plot(
            self.shap_values,
            self.X_sc_,
            feature_names=display_names,
            plot_type="dot",
            show=False,
            max_display=15,
        )
        plt.title(f"SHAP Feature Impact — {self.model_name}", pad=15)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: results/shap_summary.png")

    def plot_bar_importance(self):
        """SHAP bar chart — mean absolute impact per feature."""
        import shap
        display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in self.feature_names_]

        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            self.shap_values,
            self.X_sc_,
            feature_names=display_names,
            plot_type="bar",
            show=False,
            max_display=15,
        )
        plt.title(f"SHAP Global Feature Importance — {self.model_name}", pad=15)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: results/shap_bar_importance.png")

    def explain_candidate(self, candidate_index: int) -> dict:
        """
        Generate a human-readable explanation for one candidate.

        Returns:
            dict with top positive and negative factors,
            probability, and plain-English explanation.
        """
        if self.shap_values is None:
            raise ValueError("Call fit() first.")

        shap_row  = self.shap_values[candidate_index]
        feature_vals = self.X_sc_[candidate_index]
        prob = float(self.model.predict_proba(
            feature_vals.reshape(1, -1)
        )[0][1])

        # Top factors pushing toward shortlisting (positive SHAP)
        top_positive = []
        top_negative = []

        for i, (name, val, shap_val) in enumerate(
            zip(self.feature_names_, feature_vals, shap_row)
        ):
            display = FEATURE_DISPLAY_NAMES.get(name, name)
            entry = {
                "feature":     name,
                "display_name": display,
                "shap_value":  round(float(shap_val), 4),
                "raw_value":   round(float(val), 4),
            }
            if shap_val > 0:
                top_positive.append(entry)
            else:
                top_negative.append(entry)

        top_positive = sorted(top_positive, key=lambda x: x["shap_value"], reverse=True)[:5]
        top_negative = sorted(top_negative, key=lambda x: x["shap_value"])[:5]

        # Plain-English explanation
        pos_reasons = ", ".join(e["display_name"] for e in top_positive[:3])
        neg_reasons = ", ".join(e["display_name"] for e in top_negative[:3])

        if prob >= 0.5:
            explanation = (
                f"This candidate is recommended for shortlisting "
                f"(probability {prob:.1%}). "
                f"Key supporting factors: {pos_reasons}. "
                f"Minor concerns: {neg_reasons}."
            )
        else:
            explanation = (
                f"This candidate is not recommended for shortlisting "
                f"(probability {prob:.1%}). "
                f"Main limiting factors: {neg_reasons}. "
                f"Positive signals: {pos_reasons}."
            )

        return {
            "candidate_index":  candidate_index,
            "shortlist_prob":   round(prob, 4),
            "shortlisted":      prob >= 0.5,
            "top_positive":     top_positive,
            "top_negative":     top_negative,
            "explanation":      explanation,
        }

    def plot_waterfall(self, candidate_index: int):
        """Waterfall plot for a single candidate explanation."""
        import shap
        display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in self.feature_names_]

        explanation = shap.Explanation(
            values=self.shap_values[candidate_index],
            base_values=self.explainer.expected_value
                if not isinstance(self.explainer.expected_value, list)
                else self.explainer.expected_value[1],
            data=self.X_sc_[candidate_index],
            feature_names=display_names,
        )

        plt.figure(figsize=(9, 5))
        shap.plots.waterfall(explanation, show=False, max_display=12)
        plt.title(f"Candidate {candidate_index} — SHAP Explanation", pad=10)
        plt.tight_layout()
        out = RESULTS_DIR / f"shap_candidate_{candidate_index}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

    def run_full(self):
        """Run all SHAP analysis and save outputs."""
        self.fit()
        self.plot_global_summary()
        self.plot_bar_importance()

        # Explain a shortlisted and a non-shortlisted candidate
        preds = self.model.predict(self.X_sc_)
        shortlisted_idx     = np.where(preds == 1)[0][0]
        not_shortlisted_idx = np.where(preds == 0)[0][0]

        for idx in [shortlisted_idx, not_shortlisted_idx]:
            result = self.explain_candidate(idx)
            print(f"\n--- Candidate {idx} ---")
            print(result["explanation"])
            self.plot_waterfall(idx)

        return self


if __name__ == "__main__":
    explainer = SHAPExplainer()
    explainer.run_full()
    print("\nSHAP explainability analysis complete.")
