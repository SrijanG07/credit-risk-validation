"""Run from repository root: python -m credit_risk.pipeline."""
import argparse
import hashlib
import json
import platform
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from .data import load_data, split_data, FEATURES, TARGET, SHA256, PAY
from .metrics import evaluate, bootstrap_auc, psi
from .report import write_report

def reconcile(root, df, predictions):
    with sqlite3.connect(root / "artifacts/portfolio.sqlite") as con:
        df.to_sql("borrowers", con, if_exists="replace", index=False)
        predictions.to_sql("predictions", con, if_exists="replace", index=False)
        if predictions.duplicated(["model", "ID"]).any():
            raise ValueError("Duplicate prediction keys")
        unmatched = con.execute("SELECT COUNT(*) FROM predictions p LEFT JOIN borrowers b ON p.ID=b.ID WHERE b.ID IS NULL").fetchone()[0]
        if unmatched:
            raise ValueError("Unmatched prediction IDs")
        sql = pd.read_sql_query((root / "sql/portfolio.sql").read_text(), con)
        segments = pd.read_sql_query((root / "sql/segments.sql").read_text(), con)
        deciles = pd.read_sql_query((root / "sql/deciles.sql").read_text(), con)
        if (root / "reports").exists():
            deciles.to_csv(root / "reports/risk_deciles.csv", index=False)
        joined = predictions.merge(df[["ID", TARGET]], on="ID", validate="many_to_one")
        for row in sql.itertuples():
            a = joined[joined.model == row.model]
            expected = [len(a), a[TARGET].sum(), a[TARGET].mean(), a.probability.mean(), ((a.probability-a[TARGET])**2).mean()]
            if not np.allclose(expected, [row.n, row.defaults, row.observed_rate, row.predicted_rate, row.brier], rtol=1e-10, atol=1e-12):
                raise ValueError("SQL/Python report reconciliation failed")
        return sql, segments

def run(root, bootstrap=300):
    for d in ["reports", "artifacts"]:
        (root / d).mkdir(exist_ok=True)
    df = load_data(root)
    partitions, groups = split_data(df)
    train, val, test = [df.iloc[partitions[k]] for k in ["train", "validation", "test"]]
    tx, ty = train[FEATURES], train[TARGET]
    cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=17).split(tx, ty, groups[partitions["train"]]))
    # Fixed specifications: no test-driven search. Calibrator only sees OOF train scores.
    logistic = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=42))
    boosting = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, learning_rate=.05, l2_regularization=5, early_stopping=False, random_state=42)
    models = {"logistic": logistic, "boosting": CalibratedClassifierCV(boosting, method="sigmoid", cv=cv, ensemble=True)}
    validation, scores, probs = {}, {}, {}
    for name, model in models.items():
        model.fit(tx, ty)
        validation[name] = evaluate(val[TARGET], model.predict_proba(val[FEATURES])[:, 1])
    selected = max(validation, key=lambda k: validation[k]["roc_auc"])
    # Locked evaluation: selection above cannot read held-out test scores.
    rows = []
    for name, model in models.items():
        p = model.predict_proba(test[FEATURES])[:, 1]
        probs[name] = p
        scores[name] = evaluate(test[TARGET], p)
        scores[name]["roc_auc_ci95"] = bootstrap_auc(test[TARGET], p, groups[partitions["test"]], n=bootstrap)
        rows.append(pd.DataFrame({"ID": test.ID.to_numpy(), "model": name, "probability": p}))
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(root / "artifacts/test_predictions.csv", index=False)
    assignments = pd.DataFrame({"ID": df.ID, "split": "", "group": groups.astype(str)})
    for name, ix in partitions.items():
        assignments.loc[ix, "split"] = name
    assignments.to_csv(root / "artifacts/splits.csv", index=False)
    sql, segments = reconcile(root, df, predictions)
    sql.to_csv(root / "reports/sql_reconciliation.csv", index=False)
    segments.to_csv(root / "reports/segments.csv", index=False)
    # Monitoring is reference-vs-holdout and explicitly synthetic adverse covariate shift.
    shifted = test[FEATURES].copy()
    shifted["LIMIT_BAL"] *= .7
    shifted[PAY] = (shifted[PAY] + 1).clip(upper=9)
    chosen = models[selected]
    reference_p = chosen.predict_proba(val[FEATURES])[:, 1]
    shifted_p = chosen.predict_proba(shifted)[:, 1]
    drift = []
    for feature in FEATURES:
        drift.append({"feature": feature, "holdout_psi": psi(val[feature], test[feature]), "synthetic_psi": psi(val[feature], shifted[feature])})
    drift.append({"feature": "prediction", "holdout_psi": psi(reference_p, probs[selected]), "synthetic_psi": psi(reference_p, shifted_p)})
    drift = pd.DataFrame(drift)
    drift.to_csv(root / "reports/drift.csv", index=False)
    # Validation-only importance avoids iterative interpretation of held-out labels.
    importance = permutation_importance(chosen, val[FEATURES], val[TARGET], scoring="roc_auc", n_repeats=3, random_state=42)
    imp = pd.DataFrame({"feature": FEATURES, "auc_drop_mean": importance.importances_mean, "auc_drop_std": importance.importances_std}).sort_values("auc_drop_mean", ascending=False)
    imp.to_csv(root / "reports/importance.csv", index=False)
    audit = []
    for field in ["SEX", "AGE"]:
        labels = test[field] if field == "SEX" else pd.cut(test.AGE, [0, 30, 45, 120], labels=["<=30", "31-45", ">45"])
        for group in sorted(labels.unique(), key=str):
            mask = np.asarray(labels == group)
            y, p = test[TARGET].to_numpy()[mask], probs[selected][mask]
            audit.append({"field": field, "group": str(group), "n": int(mask.sum()), "observed_rate": float(y.mean()), "predicted_rate": float(p.mean()), "roc_auc": evaluate(y, p)["roc_auc"] if len(np.unique(y)) == 2 else None})
    pd.DataFrame(audit).to_csv(root / "reports/subgroup_audit.csv", index=False)
    summary = {"dataset_sha256": SHA256, "seed": 42, "n": len(df), "features": FEATURES, "default_rate": float(df[TARGET].mean()), "duplicate_predictor_rows": int(df.duplicated(FEATURES).sum()), "splits": {k: {"n": len(v), "default_rate": float(df.iloc[v][TARGET].mean())} for k, v in partitions.items()}, "validation": validation, "test": scores, "selected_on_validation": selected, "bootstrap_replicates": bootstrap, "sql_reconciled": True, "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}, "split_manifest_sha256": hashlib.sha256((root / "artifacts/splits.csv").read_bytes()).hexdigest()}
    (root / "reports/results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(root, summary, test[TARGET], probs, drift, imp, sql, segments)
    print(json.dumps(summary, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    if args.bootstrap < 20:
        parser.error("Use at least 20 bootstrap replicates")
    run(args.root.resolve(), args.bootstrap)

if __name__ == "__main__":
    main()
