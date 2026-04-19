"""
ml/ranking_engine.py
Trains, evaluates and saves the candidate ranking ML model.
Compares multiple classifiers and selects best performer.
Also implements the rule-based baseline for comparison.

Run: python src/ml/ranking_engine.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocessing.resume_processor import ResumeProcessor

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ─── BASELINE: Rule-based keyword filter ─────────────────────────────────────
class RuleBasedBaseline:
    """
    Traditional keyword-based recruitment filter.
    Mirrors the rule-based approach used in conventional HR systems.
    """

    REQUIRED_KEYWORDS = ["experience", "skills", "degree"]
    MIN_WORD_COUNT = 50

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        predictions = []
        for _, row in df.iterrows():
            text = str(row.get("resume_text", "")).lower()
            word_count = len(text.split())
            keyword_hits = sum(1 for kw in self.REQUIRED_KEYWORDS if kw in text)
            years = row.get("years_experience", 0)
            degree = str(row.get("degree", "")).lower()
            has_degree = degree not in ["none", ""]

            # Rule: shortlist if has degree, 2+ years experience, keywords present
            shortlist = (
                has_degree
                and years >= 2
                and keyword_hits >= 2
                and word_count >= self.MIN_WORD_COUNT
            )
            predictions.append(1 if shortlist else 0)
        return np.array(predictions)


# ─── ML RANKING ENGINE ────────────────────────────────────────────────────────
class RecruitmentRankingEngine:
    """
    Trains and evaluates multiple ML classifiers for candidate shortlisting.
    Selects the best model based on F1 score.
    """

    CANDIDATES = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42
        ),
    }

    def __init__(self):
        self.best_model_name = None
        self.best_model      = None
        self.scaler          = StandardScaler()
        self.processor       = ResumeProcessor()
        self.results_        = {}
        self.feature_names_  = None

    def load_data(self, csv_path: str):
        """Load dataset and split into features and target."""
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset: {len(df)} candidates")
        print(f"Shortlist rate: {df['shortlisted'].mean()*100:.1f}%")

        X = self.processor.process_dataframe(df)
        y = df["shortlisted"].values
        self.feature_names_ = self.processor.get_feature_names()
        self.df_raw_ = df

        return X, y, df

    def train_evaluate(self, csv_path: str = "data/AI_Resume_Screening.csv.csv"):
        """Full training and evaluation pipeline."""

        # Load and split data
        X, y, df_raw = self.load_data(csv_path)
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)

        print(f"\nTraining set: {len(X_train)} | Test set: {len(X_test)}")
        print("=" * 60)

        # ── Train and evaluate each model ────────────────────────────────────
        best_f1   = 0
        cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.CANDIDATES.items():
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]

            # Cross-validation F1
            cv_scores = cross_val_score(
                model, X_train_sc, y_train, cv=cv, scoring="f1"
            )

            metrics = {
                "precision":    round(precision_score(y_test, y_pred), 4),
                "recall":       round(recall_score(y_test, y_pred), 4),
                "f1":           round(f1_score(y_test, y_pred), 4),
                "roc_auc":      round(roc_auc_score(y_test, y_prob), 4),
                "cv_f1_mean":   round(cv_scores.mean(), 4),
                "cv_f1_std":    round(cv_scores.std(), 4),
            }
            self.results_[name] = metrics

            print(f"\n{name}")
            print(f"  Precision:  {metrics['precision']}")
            print(f"  Recall:     {metrics['recall']}")
            print(f"  F1 Score:   {metrics['f1']}")
            print(f"  ROC-AUC:    {metrics['roc_auc']}")
            print(f"  CV F1:      {metrics['cv_f1_mean']} ± {metrics['cv_f1_std']}")

            if metrics["f1"] > best_f1:
                best_f1              = metrics["f1"]
                self.best_model_name = name
                self.best_model      = model
                self.X_test_         = X_test_sc
                self.y_test_         = y_test
                self.y_pred_         = y_pred
                self.y_prob_         = y_prob
                self.idx_test_       = idx_test
                self.df_test_        = df_raw.iloc[idx_test].copy()

        print(f"\n{'='*60}")
        print(f"Best model: {self.best_model_name} (F1={best_f1})")

        # ── Baseline comparison ───────────────────────────────────────────────
        baseline = RuleBasedBaseline()
        baseline_pred = baseline.predict(self.df_test_)
        self.results_["Rule-Based Baseline"] = {
            "precision": round(precision_score(y_test, baseline_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, baseline_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, baseline_pred, zero_division=0), 4),
            "roc_auc":   "N/A",
            "cv_f1_mean":"N/A",
            "cv_f1_std": "N/A",
        }

        self._save_results()
        self._save_model()
        self._plot_all()

        return self.results_

    def _save_model(self):
        """Persist best model and scaler."""
        joblib.dump(self.best_model, MODELS_DIR / "best_model.pkl")
        joblib.dump(self.scaler,     MODELS_DIR / "scaler.pkl")
        with open(MODELS_DIR / "feature_names.json", "w") as f:
            json.dump(self.feature_names_, f, indent=2)
        with open(MODELS_DIR / "model_info.json", "w") as f:
            json.dump({
                "best_model_name": self.best_model_name,
                "metrics": self.results_[self.best_model_name]
            }, f, indent=2)
        print(f"\nModel saved to {MODELS_DIR}/")

    def _save_results(self):
        """Save comparison results to CSV."""
        rows = []
        for name, m in self.results_.items():
            rows.append({"Model": name, **m})
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / "model_comparison.csv", index=False
        )

    def _plot_all(self):
        """Generate evaluation charts."""
        self._plot_model_comparison()
        self._plot_confusion_matrix()
        self._plot_roc_curve()
        self._plot_feature_importance()

    def _plot_model_comparison(self):
        """Bar chart comparing all models."""
        models   = [k for k in self.results_ if k != "Rule-Based Baseline"]
        baseline = self.results_.get("Rule-Based Baseline", {})
        metrics  = ["precision", "recall", "f1"]

        x   = np.arange(len(models))
        w   = 0.25
        fig, ax = plt.subplots(figsize=(10, 5))

        for i, metric in enumerate(metrics):
            vals = [self.results_[m][metric] for m in models]
            ax.bar(x + i*w, vals, w, label=metric.capitalize(), alpha=0.85)
            # Baseline line
            if isinstance(baseline.get(metric), float):
                ax.axhline(y=baseline[metric], color=f"C{i}",
                           linestyle="--", alpha=0.5,
                           label=f"Baseline {metric}" if i == 0 else "")

        ax.set_xticks(x + w)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison: AI vs Rule-Based Baseline")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
        plt.close()
        print("Saved: results/model_comparison.png")

    def _plot_confusion_matrix(self):
        """Confusion matrix for best model."""
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(self.y_test_, self.y_pred_)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Not Shortlisted", "Shortlisted"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {self.best_model_name}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
        plt.close()
        print("Saved: results/confusion_matrix.png")

    def _plot_roc_curve(self):
        """ROC curve for best model."""
        fpr, tpr, _ = roc_curve(self.y_test_, self.y_prob_)
        auc = roc_auc_score(self.y_test_, self.y_prob_)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f"{self.best_model_name} (AUC={auc:.3f})")
        plt.plot([0,1],[0,1],"k--", alpha=0.4, label="Random (AUC=0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — Best Model")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
        plt.close()
        print("Saved: results/roc_curve.png")

    def _plot_feature_importance(self):
        """Feature importance for tree-based models."""
        if not hasattr(self.best_model, "feature_importances_"):
            return
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        names   = [self.feature_names_[i] for i in indices]

        plt.figure(figsize=(8, 5))
        plt.barh(names[::-1], importances[indices][::-1], color="#2196F3", alpha=0.85)
        plt.xlabel("Importance")
        plt.title(f"Top 15 Feature Importances — {self.best_model_name}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
        plt.close()
        print("Saved: results/feature_importance.png")

    def predict_candidate(self, candidate_dict: dict) -> dict:
        """Score a single candidate dict. Returns score and shortlist decision."""
        df_single = pd.DataFrame([candidate_dict])
        X = self.processor.process_dataframe(df_single)
        X_sc = self.scaler.transform(X)
        prob  = self.best_model.predict_proba(X_sc)[0][1]
        pred  = int(prob >= 0.5)
        return {
            "shortlist_probability": round(float(prob), 4),
            "shortlisted":           bool(pred),
            "confidence":            "High" if abs(prob - 0.5) > 0.3 else "Medium"
        }


if __name__ == "__main__":
    engine = RecruitmentRankingEngine()
    results = engine.train_evaluate()
    print("\n--- Final Comparison ---")
    for model, metrics in results.items():
        print(f"{model:30s}  F1={metrics['f1']}  AUC={metrics.get('roc_auc','N/A')}")
