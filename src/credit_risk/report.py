"""Portable HTML validation report with static, reproducible figures."""
import base64
import io
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

def write_report(root, summary, y, probabilities, drift, importance, sql, segments):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .15, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for (name, p), color in zip(probabilities.items(), ["#187f91", "#e27735"]):
        fpr, tpr, _ = roc_curve(y, p)
        axes[0, 0].plot(fpr, tpr, label=name, color=color, lw=2)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0, 1].plot(recall, precision, label=name, color=color, lw=2)
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        axes[1, 0].plot(predicted, observed, "o-", label=name, color=color)
        axes[1, 1].hist(p, bins=np.linspace(0,1,21), alpha=.5, label=name, color=color)
    axes[0, 0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 1].axhline(np.mean(y), linestyle="--", color="gray", label="prevalence")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray")
    for ax, title, x, ylabel in zip(axes.flat, ["Discrimination · ROC", "Precision vs recall", "Calibration · equal-count bins", "Held-out probability distribution"], ["False positive rate", "Recall", "Mean predicted probability", "Predicted probability"], ["True positive rate", "Precision", "Observed default rate", "Clients"]):
        ax.set(title=title, xlabel=x, ylabel=ylabel)
        ax.legend(fontsize=8)
    fig.savefig(root / "reports/validation.png", dpi=160)
    memory = io.BytesIO()
    fig.savefig(memory, format="png", dpi=130)
    plt.close(fig)
    encoded = base64.b64encode(memory.getvalue()).decode()
    table = pd.DataFrame(summary["test"]).T.drop(columns="roc_auc_ci95").astype(float)
    table_html = table.to_html(float_format=lambda x: f"{x:.4f}")
    winner = summary["selected_on_validation"]
    auc = summary["test"][winner]["roc_auc"]
    ci = summary["test"][winner]["roc_auc_ci95"]
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Credit Risk | Model Validation</title>
<style>body{{margin:0;background:#edf2f6;color:#172c40;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 24px}}header{{background:#142d45;color:white;padding:40px;border-radius:18px}}.eyebrow{{color:#80d5d1;letter-spacing:3px;font-size:12px}}h1{{font-size:40px;line-height:1.15;margin:14px 0}}h2{{font-size:23px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}}.card,section{{background:white;border-radius:12px;padding:24px;margin:16px 0;box-shadow:0 3px 15px #16354d08}}.value{{font-size:30px;font-weight:700;color:#127f88}}.muted{{color:#52677a}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px;border:0;border-bottom:1px solid #e2eaf0;text-align:right}}th{{background:#f3f7fa}}img{{width:100%;height:auto}}.scroll{{overflow:auto}}.note{{border-left:4px solid #e27735;padding:12px 20px;background:#fff6ee}}footer{{font-size:13px;color:#52677a}}@media(max-width:650px){{.cards{{grid-template-columns:1fr}}h1{{font-size:30px}}header{{padding:24px}}}}</style></head><body><main>
<header><div class="eyebrow">RISK ANALYTICS / INDEPENDENT PORTFOLIO STUDY</div><h1>Credit Risk Prediction<br>&amp; Model Validation</h1><p>Transparent benchmarks. Reconciled reports. Monitoring with explicit limits.</p><p>UCI Taiwan credit-card cohort · seed 42 · original Python + SQL implementation</p></header>
<div class="cards"><div class="card">Clients<div class="value">{summary['n']:,}</div><span class="muted">{summary['splits']['test']['n']:,} held out for testing</span></div><div class="card">Selected model ROC-AUC<div class="value">{auc:.3f}</div><span class="muted">95% cluster bootstrap: {ci[0]:.3f}–{ci[1]:.3f}</span></div><div class="card">Report integrity<div class="value">Reconciled</div><span class="muted">SQL and Python totals agree</span></div></div>
<section><h2>Validation decision</h2><p><b>{winner}</b> was selected by validation ROC-AUC before computing test metrics. Both fixed model specifications are reported. PR-AUC here means average precision (AP), not trapezoidal PR integration. Lower Brier, log loss and ECE are better.</p><div class="scroll">{table_html}</div><p class="note">Research benchmark only. This is a random, duplicate-grouped holdout from one historical cohort, not a temporal or external validation. It does not establish suitability for lending, Citi deployment, CCAR or CECL.</p><img alt="ROC, precision-recall, calibration and probability distribution charts" src="data:image/png;base64,{encoded}"></section>
<section><h2>Leakage controls &amp; uncertainty</h2><p>Client ID, target and demographics are excluded from predictors. Identical predictor vectors remain in the same partition. Preprocessing is fit inside the training pipeline. Boosting uses sigmoid calibration with three grouped training folds. The validation set selects the model; the test set is not used to tune it.</p><p>AUC intervals use {summary['bootstrap_replicates']} cluster-bootstrap samples of held-out predictor groups. These intervals condition on the trained model and do not capture training instability or future population shift. No decision threshold or business cost assumption is optimized.</p></section>
<section><h2>Monitoring laboratory</h2><p>Reference = validation cohort. Current = held-out cohort. Synthetic adverse scenario reduces credit limits by 30% and adds one to repayment-status codes (capped at 9). It is a feature perturbation, not an economic forecast. No outcome metric is computed for invented future labels.</p><p>PSI uses reference-fitted quantile bins, infinite tails, a missing bin and 0.5 pseudocounts. Review values above 0.10; prioritize values above 0.25. These are configurable-style investigation heuristics, not regulatory or significance thresholds.</p><div class="scroll">{drift.to_html(index=False, float_format=lambda x:f'{x:.4f}')}</div></section>
<section><h2>Independent reporting checks</h2><p>SQL joins borrower IDs to predictions and reconciles count, defaults, observed rate, mean probability and Brier score against independent pandas calculations. Failures stop report generation.</p><div class="scroll">{sql.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div><h2>Credit-limit segments</h2><div class="scroll">{segments.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div></section>
<section><h2>What drives the model?</h2><p>Validation-set permutation importance: mean ROC-AUC decrease across three shuffles. Correlated payment histories can share importance; this is not causality or an explanation of individual credit decisions.</p><div class="scroll">{importance.head(8).to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div><h2>Limitations &amp; next review</h2><ul><li>Historical Taiwanese credit-card clients do not represent current US lending or new applicants.</li><li>Labels concern next-month default; features are past account behavior. This is behavioral scoring, not an origination scorecard.</li><li>Sex and age subgroup diagnostics are supplied separately; excluding demographics does not eliminate proxies or demonstrate fairness.</li><li>Obtain newer, independently sourced temporal cohorts and business costs before considering operational use.</li></ul></section>
<footer>Data: I-Cheng Yeh (2009), UCI Default of Credit Card Clients, DOI 10.24432/C55S3H, CC BY 4.0. Source, license attribution and exact run environment are documented in the repository. No Citi affiliation.</footer></main></body></html>'''
    (root / "reports/index.html").write_text(html, encoding="utf-8")
