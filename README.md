# AI Recruitment System
### Dissertation: AI in Recruitment System
**Student:** Krishna Chaudhary | **ID:** 2912392 | **Module:** CN7000

---

## Project Structure

```
recruitment/
├── data/
│   └── AI_Resume_Screening.csv.csv       ← auto-generated dataset
├── models/
│   ├── best_model.pkl              ← trained ML model
│   ├── scaler.pkl                  ← feature scaler
│   ├── feature_names.json
│   └── model_info.json
├── results/
│   ├── model_comparison.png        ← AI vs baseline chart
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── fairness_selection_rates.png
│   ├── fairness_heatmap.png
│   ├── shap_summary.png
│   ├── shap_bar_importance.png
│   └── fairness_report.json
├── src/
│   ├── data_generator.py           ← synthetic dataset generator
│   ├── preprocessing/
│   │   └── resume_processor.py     ← NLP feature extraction
│   ├── ml/
│   │   └── ranking_engine.py       ← ML training + evaluation
│   ├── fairness/
│   │   └── fairness_evaluator.py   ← bias detection
│   └── explainability/
│       └── shap_explainer.py       ← SHAP explanations
├── dashboard/
│   └── app.py                      ← Flask web UI
├── run_pipeline.py                 ← runs everything
└── requirements.txt
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model (for NLP)
python -m spacy download en_core_web_sm
```

---

## Run Complete Pipeline

```bash
# Run all steps (data generation → training → fairness → SHAP)
python run_pipeline.py

# Then launch dashboard
python dashboard/app.py
# Open: http://localhost:5000
```

---

## Run Individual Steps

```bash
# Step 1: Generate dataset
python src/data_generator.py

# Step 2: Train models
python src/ml/ranking_engine.py

# Step 3: Fairness evaluation
python src/fairness/fairness_evaluator.py

# Step 4: SHAP explainability
python src/explainability/shap_explainer.py

# Step 5: Dashboard
python dashboard/app.py
```

---

## Using the Kaggle Dataset

If using the real Kaggle dataset instead of synthetic data:
https://www.kaggle.com/datasets/mdtalhask/ai-powered-resume-screening-dataset-2025

1. Download and save as `data/kaggle_resumes.csv`
2. Update the `csv_path` parameter in `ranking_engine.py`
3. Ensure the dataset has these columns:
   - `shortlisted` (0/1 label)
   - `resume_text` or equivalent text column
   - `years_experience`, `degree`, `gpa`, `skills`
   - `gender`, `ethnicity` (for fairness evaluation)

---

## What Each Module Does

| Module | Purpose |
|--------|---------|
| `data_generator.py` | Creates 1,000 synthetic candidate profiles with demographics |
| `resume_processor.py` | Extracts 20+ numeric features from resume text and structured fields |
| `ranking_engine.py` | Trains 4 ML models, compares against rule-based baseline, saves best |
| `fairness_evaluator.py` | Computes demographic parity, disparate impact, equal opportunity metrics |
| `shap_explainer.py` | Generates SHAP explanations for global and individual candidate decisions |
| `dashboard/app.py` | Web UI for recruiters to score candidates and view explanations |

---

## Fairness Metrics Explained

| Metric | Formula | Ideal | Threshold |
|--------|---------|-------|-----------|
| Demographic Parity Difference | max_rate − min_rate | 0 | < 0.10 |
| Disparate Impact Ratio | min_rate / max_rate | 1.0 | ≥ 0.80 (four-fifths rule) |
| Equal Opportunity Difference | max_TPR − min_TPR | 0 | < 0.10 |

---

## References

- Barocas, S. and Selbst, A. (2016) 'Big data's disparate impact', California Law Review
- Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', NeurIPS
- Mehrabi, N. et al. (2021) 'A survey on bias and fairness in machine learning', ACM Computing Surveys
- Raghavan, M. et al. (2020) 'Mitigating bias in algorithmic hiring', ACM FAccT
# AI_Recruitment_System.
