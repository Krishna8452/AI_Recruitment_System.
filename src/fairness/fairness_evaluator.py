"""
fairness/fairness_evaluator.py
Evaluates the trained model for bias and fairness across demographic groups.

Fairness metrics implemented:
  - Demographic Parity Difference
  - Equal Opportunity Difference
  - Disparate Impact Ratio
  - Selection Rate by group
  - Per-group precision, recall, F1

Reference: Barocas & Selbst (2016), Mehrabi et al. (2021)
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocessing.resume_processor import ResumeProcessor

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


class FairnessEvaluator:
    """
    Evaluates a trained classifier for demographic bias.
    Produces per-group metrics and fairness summary report.
    """

    def __init__(self):
        self.model     = joblib.load(MODELS_DIR / "best_model.pkl")
        self.scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
        self.processor = ResumeProcessor()

    def evaluate(self, csv_path: str = "data/AI_Resume_Screening.csv.csv") -> dict:
        """Run full fairness evaluation on the dataset."""
        df = pd.read_csv(csv_path)

        # Get predictions
        X      = self.processor.process_dataframe(df)
        X_sc   = self.scaler.transform(X)
        df["predicted"]    = self.model.predict(X_sc)
        df["predict_prob"] = self.model.predict_proba(X_sc)[:, 1]

        report = {}

        # ── Gender fairness ───────────────────────────────────────────────────
        report["gender"] = self._group_metrics(df, "gender", "shortlisted", "predicted")

        # ── Ethnicity fairness ────────────────────────────────────────────────
        report["ethnicity"] = self._group_metrics(df, "ethnicity", "shortlisted", "predicted")

        # ── Age group fairness ────────────────────────────────────────────────
        df["age_group"] = pd.cut(
            df["age"],
            bins=[21, 30, 40, 50, 60],
            labels=["22–30", "31–40", "41–50", "51–60"]
        )
        report["age_group"] = self._group_metrics(df, "age_group", "shortlisted", "predicted")

        # ── Fairness summary metrics ──────────────────────────────────────────
        report["fairness_summary"] = self._fairness_summary(df, report)

        # Save and plot
        self._save_report(report)
        self._plot_selection_rates(df)
        self._plot_group_metrics(report)

        return report

    def _group_metrics(self, df, group_col, y_true_col, y_pred_col) -> dict:
        """Compute per-group classification metrics."""
        results = {}
        groups = df[group_col].dropna().unique()

        for group in sorted(groups, key=str):
            mask = df[group_col] == group
            y_true = df.loc[mask, y_true_col].values
            y_pred = df.loc[mask, y_pred_col].values

            if len(y_true) < 5:
                continue

            n_total     = len(y_true)
            n_selected  = int(y_pred.sum())
            select_rate = round(float(y_pred.mean()), 4)
            true_rate   = round(float(y_true.mean()), 4)

            results[str(group)] = {
                "n":               n_total,
                "selection_rate":  select_rate,
                "true_rate":       true_rate,
                "precision":       round(precision_score(y_true, y_pred, zero_division=0), 4),
                "recall":          round(recall_score(y_true, y_pred, zero_division=0), 4),
                "f1":              round(f1_score(y_true, y_pred, zero_division=0), 4),
            }

        return results

    def _fairness_summary(self, df, report) -> dict:
        """
        Compute headline fairness metrics.

        Demographic Parity Difference (DPD):
            |P(Y_hat=1 | A=a) - P(Y_hat=1 | A=b)|
            Ideal: 0 (same selection rate for all groups)
            Threshold: < 0.10 considered acceptable

        Disparate Impact Ratio (DIR):
            min_group_rate / max_group_rate
            Ideal: 1.0 (equal rates)
            US legal standard: >= 0.80 (four-fifths rule)

        Equal Opportunity Difference (EOD):
            |TPR_a - TPR_b| across groups
            Ideal: 0 (same true positive rate)
        """
        summary = {}

        for attr in ["gender", "ethnicity"]:
            groups = report.get(attr, {})
            if len(groups) < 2:
                continue

            rates = {g: m["selection_rate"] for g, m in groups.items()}
            recalls = {g: m["recall"] for g, m in groups.items()}

            max_rate = max(rates.values())
            min_rate = min(rates.values())

            dpd = round(max_rate - min_rate, 4)
            dir_ratio = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0
            eod = round(max(recalls.values()) - min(recalls.values()), 4)

            summary[attr] = {
                "demographic_parity_difference": dpd,
                "disparate_impact_ratio":        dir_ratio,
                "equal_opportunity_difference":  eod,
                "selection_rates":               rates,
                "dpd_acceptable":                dpd < 0.10,
                "dir_acceptable":                dir_ratio >= 0.80,
                "highest_rate_group":            max(rates, key=rates.get),
                "lowest_rate_group":             min(rates, key=rates.get),
            }

            print(f"\n--- Fairness: {attr.upper()} ---")
            print(f"  Selection rates: {rates}")
            print(f"  Demographic Parity Difference: {dpd}  ({'✅ OK' if dpd < 0.10 else '⚠️  BIAS'})")
            print(f"  Disparate Impact Ratio:        {dir_ratio}  ({'✅ OK' if dir_ratio >= 0.80 else '⚠️  BIAS'})")
            print(f"  Equal Opportunity Difference:  {eod}")

        return summary

    def _save_report(self, report: dict):
        """Save full report as JSON."""
        out = RESULTS_DIR / "fairness_report.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFairness report saved to {out}")

    def _plot_selection_rates(self, df):
        """Bar chart: selection rates by gender and ethnicity."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for ax, col in zip(axes, ["gender", "ethnicity"]):
            grp = df.groupby(col)["predicted"].mean().reset_index()
            grp.columns = [col, "selection_rate"]
            true_grp = df.groupby(col)["shortlisted"].mean().reset_index()
            true_grp.columns = [col, "true_rate"]

            x = np.arange(len(grp))
            ax.bar(x - 0.2, grp["selection_rate"], 0.35,
                   label="AI Predicted Rate", color="#2196F3", alpha=0.85)
            ax.bar(x + 0.2, true_grp["true_rate"], 0.35,
                   label="Actual Shortlist Rate", color="#FF5722", alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(grp[col], rotation=20, ha="right")
            ax.set_ylabel("Selection Rate")
            ax.set_title(f"Selection Rate by {col.capitalize()}")
            ax.set_ylim(0, 0.8)
            ax.axhline(y=df["predicted"].mean(), color="grey",
                       linestyle="--", alpha=0.6, label="Overall rate")
            ax.legend(fontsize=8)

        plt.suptitle("Fairness Analysis: Selection Rates by Demographic Group",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fairness_selection_rates.png", dpi=150)
        plt.close()
        print("Saved: results/fairness_selection_rates.png")

    def _plot_group_metrics(self, report: dict):
        """Heatmap of per-group precision/recall/F1."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        for ax, attr in zip(axes, ["gender", "ethnicity"]):
            groups = report.get(attr, {})
            if not groups:
                continue
            rows = []
            for group, m in groups.items():
                rows.append({
                    "Group":     group,
                    "Precision": m["precision"],
                    "Recall":    m["recall"],
                    "F1":        m["f1"],
                    "Selection Rate": m["selection_rate"]
                })
            df_hm = pd.DataFrame(rows).set_index("Group")
            sns.heatmap(df_hm, ax=ax, annot=True, fmt=".3f",
                        cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.5)
            ax.set_title(f"Per-Group Metrics — {attr.capitalize()}")

        plt.suptitle("Fairness Metrics Heatmap", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fairness_heatmap.png", dpi=150)
        plt.close()
        print("Saved: results/fairness_heatmap.png")


if __name__ == "__main__":
    evaluator = FairnessEvaluator()
    report = evaluator.evaluate()
    print("\nFairness evaluation complete.")
    print(json.dumps(report["fairness_summary"], indent=2))
