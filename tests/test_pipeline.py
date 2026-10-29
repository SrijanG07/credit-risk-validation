"""Exercise the real training/report workflow without a network dependency.

Synthetic data here tests software integration; it is never used for reported
portfolio performance, which comes from the checksum-verified UCI dataset.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk import pipeline
from credit_risk.data import SCHEMA, TARGET, FEATURES, PAY, validate


@pytest.fixture
def reporting_root(tmp_path):
    (tmp_path / "artifacts").mkdir()
    shutil.copytree(Path(__file__).parents[1] / "sql", tmp_path / "sql")
    return tmp_path


def test_end_to_end_training_and_report(reporting_root, monkeypatch):
    rng = np.random.default_rng(19)
    n = 400
    frame = pd.DataFrame(rng.integers(0, 10000, (n, len(SCHEMA))), columns=SCHEMA)
    frame["ID"] = np.arange(1, n + 1)
    frame["LIMIT_BAL"] = rng.integers(1, 40, n) * 10000
    frame["SEX"] = rng.integers(1, 3, n)
    frame["AGE"] = rng.integers(21, 70, n)
    frame["EDUCATION"] = 2
    frame["MARRIAGE"] = 1
    frame[PAY] = rng.integers(-2, 5, (n, len(PAY)))
    frame[TARGET] = rng.binomial(1, .4, n)
    validate(frame)
    monkeypatch.setattr(pipeline, "load_data", lambda root: frame)
    pipeline.run(reporting_root, bootstrap=40)

    results = json.loads((reporting_root / "reports/results.json").read_text())
    assert results["n"] == n and results["sql_reconciled"] is True
    assert results["selected_on_validation"] == max(
        results["validation"], key=lambda name: results["validation"][name]["roc_auc"]
    )
    assignments = pd.read_csv(reporting_root / "artifacts/splits.csv")
    predictions = pd.read_csv(reporting_root / "artifacts/test_predictions.csv")
    expected = set(assignments.loc[assignments.split == "test", "ID"])
    for _, model_predictions in predictions.groupby("model"):
        assert set(model_predictions.ID) == expected
        assert model_predictions.probability.between(0, 1).all()
    assert assignments.groupby("group").split.nunique().max() == 1
    assert len(assignments) == n and assignments.ID.is_unique
    for name in ["index.html", "validation.png", "drift.csv", "risk_deciles.csv", "subgroup_audit.csv"]:
        assert (reporting_root / "reports" / name).stat().st_size > 0
    deciles = pd.read_csv(reporting_root / "reports/risk_deciles.csv")
    assert (deciles.groupby("model").n.sum() == len(expected)).all()
    assert set(results["features"]) == set(FEATURES)


@pytest.mark.parametrize("defect", ["dropped_row", "missing_model", "duplicate", "nan", "out_of_range", "empty"])
def test_reporting_rejects_incomplete_or_invalid_predictions(reporting_root, defect):
    borrowers = pd.DataFrame({"ID": [1, 2], "default": [0, 1], "LIMIT_BAL": [50000, 400000]})
    predictions = pd.DataFrame({"ID": [1, 2, 1, 2], "model": ["a", "a", "b", "b"], "probability": [.1, .9, .2, .8]})
    if defect == "dropped_row":
        predictions = predictions.iloc[1:]
    elif defect == "missing_model":
        predictions = predictions[predictions.model == "a"]
    elif defect == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[:1]])
    elif defect == "nan":
        predictions.loc[0, "probability"] = np.nan
    elif defect == "out_of_range":
        predictions.loc[0, "probability"] = 1.2
    elif defect == "empty":
        predictions = predictions.iloc[:0]
    with pytest.raises(ValueError):
        pipeline.reconcile(reporting_root, borrowers, predictions, expected_ids=[1, 2], expected_models=["a", "b"])


@pytest.mark.parametrize("labels,scores", [([], []), ([1, 1], [.1, .9]), ([0, 2], [.1, .9]), ([[0], [1]], [.1, .9]), ([0, 1], [[.1], [.9]])])
def test_metric_input_contract(labels, scores):
    from credit_risk.metrics import evaluate
    with pytest.raises(ValueError):
        evaluate(labels, scores)
