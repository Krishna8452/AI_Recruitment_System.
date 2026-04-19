"""
dashboard/app.py
Flask web dashboard for the AI recruitment system.
Provides REST API endpoints consumed by the recruiter frontend.

Endpoints:
  GET  /                       → Dashboard HTML
  POST /api/predict            → Score a candidate
  POST /api/explain            → SHAP explanation for a candidate
  GET  /api/model-info         → Current model metrics
  GET  /api/fairness-summary   → Fairness report summary
  GET  /api/candidates         → Dataset overview stats

Run: python dashboard/app.py
Open: http://localhost:5000
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing.resume_processor import ResumeProcessor

app = Flask(__name__)
CORS(app)

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")

# ─── Load model on startup ────────────────────────────────────────────────────
model     = joblib.load(MODELS_DIR / "best_model.pkl")
scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
processor = ResumeProcessor()

with open(MODELS_DIR / "model_info.json") as f:
    model_info = json.load(f)

with open(MODELS_DIR / "feature_names.json") as f:
    feature_names = json.load(f)


# ─── HTML DASHBOARD ───────────────────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Recruitment System — Recruiter Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f9; color: #333; }
  header { background: #1F3A52; color: white; padding: 18px 32px;
           display: flex; align-items: center; justify-content: space-between; }
  header h1 { font-size: 1.4rem; }
  header span { font-size: 0.85rem; opacity: 0.8; }
  .container { max-width: 1100px; margin: 28px auto; padding: 0 20px; }
  .card { background: white; border-radius: 8px; padding: 24px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }
  .card h2 { font-size: 1.1rem; color: #1F3A52; margin-bottom: 16px;
             border-bottom: 2px solid #e8edf2; padding-bottom: 8px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
  .stat-box { background: #f0f4f8; border-radius: 6px; padding: 16px; text-align: center; }
  .stat-box .val { font-size: 2rem; font-weight: bold; color: #1F3A52; }
  .stat-box .lbl { font-size: 0.8rem; color: #666; margin-top: 4px; }
  label { font-size: 0.85rem; color: #555; display: block; margin-bottom: 4px; }
  input, select, textarea {
    width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 5px;
    font-size: 0.9rem; margin-bottom: 14px;
  }
  textarea { height: 90px; resize: vertical; }
  button {
    background: #1F3A52; color: white; padding: 10px 24px; border: none;
    border-radius: 5px; cursor: pointer; font-size: 0.95rem; width: 100%;
  }
  button:hover { background: #2E5878; }
  #result-box { display: none; margin-top: 18px; padding: 16px; border-radius: 6px; }
  .shortlisted   { background: #e8f5e9; border-left: 4px solid #4CAF50; }
  .not-shortlisted { background: #fce4ec; border-left: 4px solid #F44336; }
  .result-title { font-weight: bold; font-size: 1.1rem; margin-bottom: 8px; }
  .prob-bar { background: #e0e0e0; border-radius: 10px; height: 10px; margin: 8px 0; }
  .prob-fill { height: 10px; border-radius: 10px; background: #4CAF50; }
  .explanation { font-size: 0.9rem; color: #444; margin-top: 10px; line-height: 1.5; }
  .factor-list { margin-top: 10px; }
  .factor { display: flex; justify-content: space-between; padding: 4px 0;
            font-size: 0.85rem; border-bottom: 1px solid #f0f0f0; }
  .positive { color: #388E3C; }
  .negative { color: #D32F2F; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
           font-size: 0.75rem; font-weight: bold; }
  .badge-ok   { background: #e8f5e9; color: #2E7D32; }
  .badge-warn { background: #fff3e0; color: #E65100; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { background: #1F3A52; color: white; padding: 8px 12px; text-align: left; }
  td { padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }
  tr:hover td { background: #f9f9f9; }
</style>
</head>
<body>
<header>
  <h1>🤖 AI Recruitment System</h1>
  <span>Explainable · Fair · Transparent</span>
</header>

<div class="container">

  <!-- Model Stats -->
  <div class="card" id="stats-card">
    <h2>Model Performance Overview</h2>
    <div class="grid-4" id="stats-grid">
      <div class="stat-box"><div class="val" id="stat-f1">—</div><div class="lbl">F1 Score</div></div>
      <div class="stat-box"><div class="val" id="stat-auc">—</div><div class="lbl">ROC-AUC</div></div>
      <div class="stat-box"><div class="val" id="stat-precision">—</div><div class="lbl">Precision</div></div>
      <div class="stat-box"><div class="val" id="stat-recall">—</div><div class="lbl">Recall</div></div>
    </div>
    <p style="font-size:0.82rem;color:#888;margin-top:12px;" id="model-name-label"></p>
  </div>

  <div class="grid-2">

    <!-- Candidate Scorer -->
    <div class="card">
      <h2>Score a Candidate</h2>
      <label>Years of Experience</label>
      <input type="number" id="years_exp" min="0" max="30" value="3">
      <label>Highest Degree</label>
      <select id="degree">
        <option>Bachelor</option>
        <option>Master</option>
        <option>PhD</option>
        <option>Diploma</option>
        <option>None</option>
      </select>
      <label>GPA (0.0 – 4.0)</label>
      <input type="number" id="gpa" step="0.1" min="0" max="4" value="3.2">
      <label>Number of Skills Listed</label>
      <input type="number" id="num_skills" min="0" max="20" value="7">
      <label>Previous Employers</label>
      <input type="number" id="prev_companies" min="0" max="10" value="2">
      <label>Skills (pipe-separated, e.g. Python|SQL|Machine Learning)</label>
      <input type="text" id="skills" value="Python|Machine Learning|SQL">
      <label>Resume Text (paste CV summary)</label>
      <textarea id="resume_text">Experienced professional with 3 years of experience. Holds a Bachelor in Computer Science. Key skills include: Python, Machine Learning, SQL.</textarea>
      <button onclick="scoreCandidate()">Score Candidate</button>

      <div id="result-box">
        <div class="result-title" id="result-title"></div>
        <div style="font-size:0.9rem;margin-bottom:6px;">
          Shortlisting Probability: <strong id="result-prob"></strong>
        </div>
        <div class="prob-bar">
          <div class="prob-fill" id="prob-bar-fill" style="width:0%"></div>
        </div>
        <div class="explanation" id="result-explanation"></div>
        <div class="factor-list" id="factor-list"></div>
      </div>
    </div>

    <!-- Fairness Panel -->
    <div class="card">
      <h2>Fairness Summary</h2>
      <div id="fairness-content">
        <p style="color:#888;font-size:0.9rem;">Loading fairness data...</p>
      </div>
    </div>

  </div>

  <!-- Results Table -->
  <div class="card">
    <h2>Dataset Overview</h2>
    <div id="dataset-stats" style="color:#888;font-size:0.9rem;">Loading...</div>
  </div>

</div>

<script>
async function loadStats() {
  try {
    const r = await fetch('/api/model-info');
    const d = await r.json();
    document.getElementById('stat-f1').textContent = d.metrics?.f1 ?? '—';
    document.getElementById('stat-auc').textContent = d.metrics?.roc_auc ?? '—';
    document.getElementById('stat-precision').textContent = d.metrics?.precision ?? '—';
    document.getElementById('stat-recall').textContent = d.metrics?.recall ?? '—';
    document.getElementById('model-name-label').textContent = 'Best model: ' + (d.best_model_name ?? '');
  } catch(e) { console.error(e); }
}

async function loadFairness() {
  try {
    const r = await fetch('/api/fairness-summary');
    const d = await r.json();
    let html = '';
    for (const [attr, data] of Object.entries(d)) {
      const dpd = data.demographic_parity_difference;
      const dir = data.disparate_impact_ratio;
      html += `<div style="margin-bottom:14px;">
        <strong>${attr.charAt(0).toUpperCase()+attr.slice(1)}</strong>
        <table style="margin-top:6px;">
          <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
          <tr><td>Demographic Parity Difference</td><td>${dpd}</td>
            <td><span class="badge ${data.dpd_acceptable?'badge-ok':'badge-warn'}">
              ${data.dpd_acceptable?'✅ OK':'⚠️ Review'}</span></td></tr>
          <tr><td>Disparate Impact Ratio</td><td>${dir}</td>
            <td><span class="badge ${data.dir_acceptable?'badge-ok':'badge-warn'}">
              ${data.dir_acceptable?'✅ OK':'⚠️ Review'}</span></td></tr>
          <tr><td>Equal Opportunity Difference</td><td>${data.equal_opportunity_difference}</td><td>—</td></tr>
        </table>
        <div style="font-size:0.8rem;color:#666;margin-top:6px;">
          Highest selection rate: ${data.highest_rate_group} |
          Lowest: ${data.lowest_rate_group}
        </div>
      </div>`;
    }
    document.getElementById('fairness-content').innerHTML = html || '<p>No fairness data available.</p>';
  } catch(e) {
    document.getElementById('fairness-content').innerHTML = '<p style="color:red">Run fairness evaluation first.</p>';
  }
}

async function loadDatasetStats() {
  try {
    const r = await fetch('/api/candidates');
    const d = await r.json();
    document.getElementById('dataset-stats').innerHTML =
      `<p>Total candidates: <strong>${d.total}</strong> | 
       Shortlist rate: <strong>${d.shortlist_rate}</strong> | 
       Avg experience: <strong>${d.avg_experience} years</strong> |
       Avg GPA: <strong>${d.avg_gpa}</strong></p>`;
  } catch(e) {}
}

async function scoreCandidate() {
  const payload = {
    years_experience: parseFloat(document.getElementById('years_exp').value),
    degree:           document.getElementById('degree').value,
    gpa:              parseFloat(document.getElementById('gpa').value),
    num_skills:       parseInt(document.getElementById('num_skills').value),
    prev_companies:   parseInt(document.getElementById('prev_companies').value),
    skills:           document.getElementById('skills').value,
    resume_text:      document.getElementById('resume_text').value,
  };

  try {
    const r = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    const box = document.getElementById('result-box');
    box.style.display = 'block';
    box.className = d.shortlisted ? 'shortlisted' : 'not-shortlisted';
    document.getElementById('result-title').textContent =
      d.shortlisted ? '✅ Recommended for Shortlisting' : '❌ Not Recommended';
    const pct = Math.round(d.shortlist_probability * 100);
    document.getElementById('result-prob').textContent = pct + '%';
    document.getElementById('prob-bar-fill').style.width = pct + '%';
    document.getElementById('prob-bar-fill').style.background = d.shortlisted ? '#4CAF50' : '#F44336';
    document.getElementById('result-explanation').textContent = d.explanation || '';

    let factorsHtml = '<div style="margin-top:10px;font-weight:bold;font-size:0.85rem;">Key Factors:</div>';
    (d.top_positive || []).slice(0,3).forEach(f => {
      factorsHtml += `<div class="factor"><span class="positive">▲ ${f.display_name}</span><span>+${f.shap_value.toFixed(3)}</span></div>`;
    });
    (d.top_negative || []).slice(0,3).forEach(f => {
      factorsHtml += `<div class="factor"><span class="negative">▼ ${f.display_name}</span><span>${f.shap_value.toFixed(3)}</span></div>`;
    });
    document.getElementById('factor-list').innerHTML = factorsHtml;
  } catch(e) {
    alert('Error: ' + e.message);
  }
}

window.onload = () => { loadStats(); loadFairness(); loadDatasetStats(); };
</script>
</body>
</html>
"""


# ─── API ROUTES ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Score a candidate and return SHAP explanation."""
    try:
        data = request.get_json(force=True)

        # Build dataframe for processing
        df_input = pd.DataFrame([{
            "years_experience": data.get("years_experience", 0),
            "degree":           data.get("degree", "Bachelor"),
            "gpa":              data.get("gpa", 3.0),
            "num_skills":       data.get("num_skills", 5),
            "prev_companies":   data.get("prev_companies", 1),
            "skills":           data.get("skills", ""),
            "resume_text":      data.get("resume_text", ""),
        }])

        X     = processor.process_dataframe(df_input)
        X_sc  = scaler.transform(X)
        prob  = float(model.predict_proba(X_sc)[0][1])
        pred  = prob >= 0.5

        # SHAP explanation for this candidate
        explanation_text = ""
        top_positive = []
        top_negative = []

        try:
            import shap
            model_type = type(model).__name__
            if model_type in ["RandomForestClassifier", "GradientBoostingClassifier",
                              "DecisionTreeClassifier"]:
                exp = shap.TreeExplainer(model)
                sv  = exp.shap_values(X_sc)
                shap_row = sv[1][0] if isinstance(sv, list) else sv[0]
            else:
                exp = shap.LinearExplainer(model, X_sc)
                shap_row = exp.shap_values(X_sc)[0]

            from src.explainability.shap_explainer import FEATURE_DISPLAY_NAMES
            fn = processor.get_feature_names()

            for name, val, sv_val in zip(fn, X_sc[0], shap_row):
                display = FEATURE_DISPLAY_NAMES.get(name, name)
                entry = {
                    "feature": name, "display_name": display,
                    "shap_value": round(float(sv_val), 4),
                    "raw_value":  round(float(val), 4),
                }
                if sv_val > 0:
                    top_positive.append(entry)
                else:
                    top_negative.append(entry)

            top_positive = sorted(top_positive, key=lambda x: x["shap_value"], reverse=True)[:5]
            top_negative = sorted(top_negative, key=lambda x: x["shap_value"])[:5]

            pos_names = ", ".join(e["display_name"] for e in top_positive[:3])
            neg_names = ", ".join(e["display_name"] for e in top_negative[:3])

            if pred:
                explanation_text = (
                    f"This candidate is recommended ({prob:.1%} probability). "
                    f"Key strengths: {pos_names}. "
                    f"Areas of concern: {neg_names}."
                )
            else:
                explanation_text = (
                    f"This candidate is not recommended ({prob:.1%} probability). "
                    f"Limiting factors: {neg_names}. "
                    f"Positive signals: {pos_names}."
                )
        except Exception as shap_err:
            explanation_text = (
                f"Candidate {'recommended' if pred else 'not recommended'} "
                f"with {prob:.1%} probability."
            )

        return jsonify({
            "shortlist_probability": round(prob, 4),
            "shortlisted":           bool(pred),
            "confidence":            "High" if abs(prob - 0.5) > 0.3 else "Medium",
            "explanation":           explanation_text,
            "top_positive":          top_positive,
            "top_negative":          top_negative,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-info")
def model_info_route():
    return jsonify(model_info)


@app.route("/api/fairness-summary")
def fairness_summary():
    report_path = RESULTS_DIR / "fairness_report.json"
    if not report_path.exists():
        return jsonify({"error": "Run fairness evaluation first: python src/fairness/fairness_evaluator.py"}), 404
    with open(report_path) as f:
        report = json.load(f)
    return jsonify(report.get("fairness_summary", {}))


@app.route("/api/candidates")
def candidates_stats():
    try:
        df = pd.read_csv("data/AI_Resume_Screening.csv.csv")
        return jsonify({
            "total":          len(df),
            "shortlist_rate": f"{df['shortlisted'].mean()*100:.1f}%",
            "avg_experience": round(df["years_experience"].mean(), 1),
            "avg_gpa":        round(df["gpa"].mean(), 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n AI Recruitment Dashboard")
    print("=" * 40)
    print("Open: http://localhost:5000")
    print("=" * 40)
    app.run(debug=True, host="0.0.0.0", port=5000)
