"""
run_pipeline.py
Master script — runs the complete AI recruitment pipeline end to end.

Steps:
  1. Generate synthetic dataset
  2. Train and evaluate ML models
  3. Fairness evaluation
  4. SHAP explainability analysis
  5. Print final summary

Usage:
  python run_pipeline.py
  python run_pipeline.py --skip-data   (if data already exists)
  python run_pipeline.py --dashboard   (also launch web UI)
"""

import argparse
import sys
import time
from pathlib import Path

def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def step1_generate_data(skip: bool):
    banner("STEP 1: Generate Synthetic Dataset")
    if skip and Path("data/AI_Resume_Screening.csv.csv").exists():
        print("Dataset already exists — skipping generation.")
        return
    from src.data_generator import generate_dataset
    df = generate_dataset(1000)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/AI_Resume_Screening.csv.csv", index=False)
    print(f"Dataset saved: data/AI_Resume_Screening.csv.csv ({len(df)} rows)")


def step2_train_model():
    banner("STEP 2: Train and Evaluate ML Models")
    from src.ml.ranking_engine import RecruitmentRankingEngine
    engine = RecruitmentRankingEngine()
    results = engine.train_evaluate()
    print("\nComparison Summary:")
    print(f"{'Model':<30} {'F1':>6} {'AUC':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 66)
    for name, m in results.items():
        f1  = m.get("f1", "—")
        auc = m.get("roc_auc", "—")
        pr  = m.get("precision", "—")
        re  = m.get("recall", "—")
        print(f"{name:<30} {str(f1):>6} {str(auc):>8} {str(pr):>10} {str(re):>8}")
    return engine


def step3_fairness():
    banner("STEP 3: Fairness Evaluation")
    from src.fairness.fairness_evaluator import FairnessEvaluator
    evaluator = FairnessEvaluator()
    report = evaluator.evaluate()
    summary = report.get("fairness_summary", {})
    for attr, data in summary.items():
        dpd = data["demographic_parity_difference"]
        dir_r = data["disparate_impact_ratio"]
        print(f"\n  {attr.upper()}:")
        print(f"    Demographic Parity Difference: {dpd}  {'✅' if data['dpd_acceptable'] else '⚠️'}")
        print(f"    Disparate Impact Ratio:        {dir_r}  {'✅' if data['dir_acceptable'] else '⚠️'}")


def step4_explain():
    banner("STEP 4: SHAP Explainability Analysis")
    try:
        import shap
        from src.explainability.shap_explainer import SHAPExplainer
        explainer = SHAPExplainer()
        explainer.run_full()
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
    except Exception as e:
        print(f"SHAP analysis skipped: {e}")


def step5_summary():
    banner("STEP 5: Final Summary")
    import json
    try:
        with open("models/model_info.json") as f:
            info = json.load(f)
        m = info["metrics"]
        print(f"\n  Best Model:  {info['best_model_name']}")
        print(f"  F1 Score:    {m['f1']}")
        print(f"  ROC-AUC:     {m['roc_auc']}")
        print(f"  Precision:   {m['precision']}")
        print(f"  Recall:      {m['recall']}")
    except Exception as e:
        print(f"Could not load model info: {e}")

    print("\n  Output files generated:")
    results_dir = Path("results")
    if results_dir.exists():
        for f in sorted(results_dir.iterdir()):
            print(f"    {f}")

    print("\n  To launch dashboard:")
    print("    python dashboard/app.py")
    print("    Open: http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Recruitment Pipeline")
    parser.add_argument("--skip-data",  action="store_true", help="Skip data generation")
    parser.add_argument("--dashboard",  action="store_true", help="Launch dashboard after pipeline")
    args = parser.parse_args()

    start = time.time()

    step1_generate_data(skip=args.skip_data)
    step2_train_model()
    step3_fairness()
    step4_explain()
    step5_summary()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f} seconds")
    print(f"{'='*60}")

    if args.dashboard:
        import subprocess
        subprocess.run([sys.executable, "dashboard/app.py"])
