"""CodeAlpha Data Science Internship — Flask dashboard.

Serves both internship tasks (Iris Classification & Sales Prediction)
on a single web UI so the project is visible in the browser preview.
"""
import base64
import os
import subprocess
import sys

import pandas as pd
from flask import Flask, render_template_string

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.join(BASE_DIR, "Task1_Iris_Classification")
TASK4_DIR = os.path.join(BASE_DIR, "Task4_Sales_Prediction")


def img_to_b64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def csv_to_html(path):
    if not os.path.exists(path):
        return '<p class="warn">Results not found. Click <strong>Re-run Analysis</strong> to generate them.</p>'
    df = pd.read_csv(path)
    return df.to_html(index=False, classes="data-table", border=0, justify="center")


def run_script(script_path):
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, cwd=os.path.dirname(script_path),
    )
    return result.stdout + ("\n" + result.stderr if result.stderr else "")


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CodeAlpha Data Science</title>
<style>
  :root { --accent:#4c72b0; --bg:#f7f8fa; --card:#fff; --text:#222; --muted:#666; --radius:10px; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }
  header { background:var(--accent); color:#fff; padding:2rem 1.5rem; text-align:center; }
  header h1 { font-size:1.8rem; margin-bottom:.3rem; }
  header p { opacity:.85; font-size:1rem; }
  nav { display:flex; justify-content:center; gap:.5rem; padding:1rem; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.08); flex-wrap:wrap; }
  nav a { padding:.5rem 1.2rem; border-radius:8px; text-decoration:none; color:var(--text); font-weight:500; transition:background .2s; }
  nav a:hover { background:#eef; }
  nav a.active { background:var(--accent); color:#fff; }
  main { max-width:1000px; margin:1.5rem auto; padding:0 1rem; }
  .card { background:var(--card); border-radius:var(--radius); padding:1.5rem; margin-bottom:1.5rem; box-shadow:0 1px 4px rgba(0,0,0,.06); }
  .card h2 { color:var(--accent); margin-bottom:.8rem; font-size:1.3rem; }
  .card h3 { margin:1rem 0 .5rem; font-size:1.05rem; }
  .card p, .card li { margin-bottom:.5rem; }
  .card ul { margin-left:1.5rem; }
  .warn { color:#c44e52; }
  .data-table { border-collapse:collapse; width:100%; margin:.5rem 0; }
  .data-table th, .data-table td { padding:.5rem .8rem; text-align:center; border-bottom:1px solid #eee; }
  .data-table th { background:#f0f2f6; font-weight:600; }
  .data-table tr:hover { background:#f8f9ff; }
  .plots { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:1rem 0; }
  .plots img { width:100%; border-radius:8px; border:1px solid #eee; }
  .full-plot img { width:70%; max-width:500px; display:block; margin:0 auto; border-radius:8px; border:1px solid #eee; }
  .btn { display:inline-block; padding:.6rem 1.4rem; background:var(--accent); color:#fff; border:none; border-radius:8px; font-size:1rem; cursor:pointer; text-decoration:none; }
  .btn:hover { opacity:.9; }
  pre { background:#1e1e2e; color:#cdd6f4; padding:1rem; border-radius:8px; overflow-x:auto; font-size:.85rem; margin:1rem 0; white-space:pre-wrap; }
  footer { text-align:center; padding:1.5rem; color:var(--muted); font-size:.85rem; }
  @media(max-width:640px){ .plots{ grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>📊 CodeAlpha Data Science Internship</h1>
  <p>Two machine-learning tasks — explore results and re-run analyses live</p>
</header>
<nav>
  <a href="/" class="{{'active' if task=='1' else ''}}">🌸 Task 1: Iris Classification</a>
  <a href="/task4" class="{{'active' if task=='4' else ''}}">📈 Task 4: Sales Prediction</a>
</nav>
<main>
  {% block content %}{% endblock %}
</main>
<footer>CodeAlpha Data Science Internship • Tasks 1 &amp; 4</footer>
</body>
</html>"""

TASK1_PAGE = """{% extends "_base.html" %}
{% block content %}
<div class="card">
  <h2>🌸 Task 1: Iris Flower Classification</h2>
  <p>Classifies Iris flowers (setosa, versicolor, virginica) from sepal and petal
     measurements using scikit-learn. Five models are compared with 5-fold
     cross-validation; the best is evaluated on a held-out test set.</p>
  <form action="/run/task1" method="post"><button class="btn" type="submit">🔄 Re-run Analysis</button></form>
  {% if output %}<h3>Script Output</h3><pre>{{ output }}</pre>{% endif %}
</div>
<div class="card">
  <h3>Model Comparison</h3>
  {{ table|safe }}
</div>
<div class="card">
  <h3>Plots</h3>
  <div class="plots">
    <div><img src="data:image/png;base64,{{ pairplot }}" alt="Pair Plot"></div>
    <div><img src="data:image/png;base64,{{ heatmap }}" alt="Correlation Heatmap"></div>
  </div>
  <div class="full-plot"><img src="data:image/png;base64,{{ confusion }}" alt="Confusion Matrix"></div>
</div>
{% endblock %}"""

TASK4_PAGE = """{% extends "_base.html" %}
{% block content %}
<div class="card">
  <h2>📈 Task 4: Sales Prediction using Python</h2>
  <p>Predicts product sales from advertising spend across three channels
     (TV, Radio, Newspaper). Four regressors are compared; the best model
     is turned into marketing insights.</p>
  <form action="/run/task4" method="post"><button class="btn" type="submit">🔄 Re-run Analysis</button></form>
  {% if output %}<h3>Script Output</h3><pre>{{ output }}</pre>{% endif %}
</div>
<div class="card">
  <h3>Model Comparison</h3>
  {{ table|safe }}
</div>
<div class="card">
  <h3>Plots</h3>
  <div class="plots">
    <div><img src="data:image/png;base64,{{ pairplot }}" alt="Pair Plot"></div>
    <div><img src="data:image/png;base64,{{ heatmap }}" alt="Correlation Heatmap"></div>
  </div>
  <div class="plots">
    <div><img src="data:image/png;base64,{{ avp }}" alt="Actual vs Predicted"></div>
    <div><img src="data:image/png;base64,{{ channel }}" alt="Channel Impact"></div>
  </div>
</div>
<div class="card">
  <h3>Business Insights</h3>
  <ul>
    <li><strong>Radio</strong> gives the highest return per unit spent (+0.19 sales per unit).</li>
    <li><strong>TV</strong> is the largest overall driver (61% feature importance) due to spend volume.</li>
    <li><strong>Newspaper</strong> has essentially no effect once TV and Radio are accounted for — best candidate to reallocate budget to Radio/TV.</li>
    <li>Tree models outperform linear ones, suggesting non-linear / interaction effects.</li>
  </ul>
</div>
{% endblock %}"""

# Write base template to a file so Jinja extends works
BASE_TEMPLATE = PAGE


@app.route("/")
def task1():
    return render_template_string(
        TASK1_PAGE,
        task="1",
        table=csv_to_html(os.path.join(TASK1_DIR, "model_comparison.csv")),
        pairplot=img_to_b64(os.path.join(TASK1_DIR, "pairplot.png")) or "",
        heatmap=img_to_b64(os.path.join(TASK1_DIR, "correlation_heatmap.png")) or "",
        confusion=img_to_b64(os.path.join(TASK1_DIR, "confusion_matrix.png")) or "",
        output="",
    )


@app.route("/task4")
def task4():
    return render_template_string(
        TASK4_PAGE,
        task="4",
        table=csv_to_html(os.path.join(TASK4_DIR, "model_comparison.csv")),
        pairplot=img_to_b64(os.path.join(TASK4_DIR, "pairplot.png")) or "",
        heatmap=img_to_b64(os.path.join(TASK4_DIR, "correlation_heatmap.png")) or "",
        avp=img_to_b64(os.path.join(TASK4_DIR, "actual_vs_predicted.png")) or "",
        channel=img_to_b64(os.path.join(TASK4_DIR, "channel_impact.png")) or "",
        output="",
    )


@app.route("/run/task1", methods=["POST"])
def run_task1():
    output = run_script(os.path.join(TASK1_DIR, "iris_classification.py"))
    return render_template_string(
        TASK1_PAGE,
        task="1",
        table=csv_to_html(os.path.join(TASK1_DIR, "model_comparison.csv")),
        pairplot=img_to_b64(os.path.join(TASK1_DIR, "pairplot.png")) or "",
        heatmap=img_to_b64(os.path.join(TASK1_DIR, "correlation_heatmap.png")) or "",
        confusion=img_to_b64(os.path.join(TASK1_DIR, "confusion_matrix.png")) or "",
        output=output,
    )


@app.route("/run/task4", methods=["POST"])
def run_task4():
    output = run_script(os.path.join(TASK4_DIR, "sales_prediction.py"))
    return render_template_string(
        TASK4_PAGE,
        task="4",
        table=csv_to_html(os.path.join(TASK4_DIR, "model_comparison.csv")),
        pairplot=img_to_b64(os.path.join(TASK4_DIR, "pairplot.png")) or "",
        heatmap=img_to_b64(os.path.join(TASK4_DIR, "correlation_heatmap.png")) or "",
        avp=img_to_b64(os.path.join(TASK4_DIR, "actual_vs_predicted.png")) or "",
        channel=img_to_b64(os.path.join(TASK4_DIR, "channel_impact.png")) or "",
        output=output,
    )


# Jinja needs the base template as a file for {% extends %}
_base_path = os.path.join(BASE_DIR, "templates")
os.makedirs(_base_path, exist_ok=True)
with open(os.path.join(_base_path, "_base.html"), "w") as f:
    f.write(PAGE)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
