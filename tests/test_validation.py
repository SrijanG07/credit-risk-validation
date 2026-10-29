import numpy as np
import pandas as pd
import pytest
from credit_risk.metrics import evaluate, psi
from credit_risk.data import FEATURES, SCHEMA, split_data, validate

def test_known_perfect_metrics():
    m = evaluate([0, 0, 1, 1], [.1, .2, .8, .9])
    assert m["roc_auc"] == m["pr_auc_ap"] == m["ks"] == 1
    assert m["brier"] == pytest.approx(.025)

def test_psi_identity_shift_and_tails():
    x = np.arange(1000)
    assert psi(x, x) == pytest.approx(0)
    assert psi(x, x + 10000) > 1
    assert psi(np.ones(100), np.ones(100) * 2) > 1
    assert np.isfinite(psi([1, 2, np.nan], [np.nan, 100, 200]))
    assert psi(np.ones(100), np.full(100, np.nan)) > 1

def test_psi_empty_rejected():
    with pytest.raises(ValueError):
        psi([], [1])

def test_probabilities_rejected():
    with pytest.raises(ValueError):
        evaluate([0, 1], [.1, 1.2])

def test_duplicate_groups_never_cross_splits():
    rng = np.random.default_rng(4)
    x = pd.DataFrame(rng.integers(1, 100, (100, len(FEATURES))), columns=FEATURES)
    x = pd.concat([x, x], ignore_index=True)
    x["default"] = np.tile([0, 1], 100)
    parts, groups = split_data(x)
    group_sets = [set(groups[v]) for v in parts.values()]
    assert sum(map(len, parts.values())) == len(x)
    for i, a in enumerate(group_sets):
        for b in group_sets[i+1:]:
            assert not a & b
    again, _ = split_data(x)
    assert all(np.array_equal(parts[k], again[k]) for k in parts)

def test_feature_contract_excludes_leakage_and_demographics():
    assert not set(FEATURES) & {"ID", "default", "SEX", "AGE", "EDUCATION", "MARRIAGE"}

def test_schema_fails_closed():
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"ID": [1]}))

def test_preprocessor_is_fit_on_training_only():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression())
    model.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    before = model.named_steps["standardscaler"].mean_.copy()
    model.predict_proba([[1000000]])
    assert np.array_equal(before, [1.5])
    assert np.array_equal(model.named_steps["standardscaler"].mean_, before)

def test_dataset_contract_rejects_duplicate_ids_and_bad_targets():
    frame = pd.DataFrame(np.ones((2, len(SCHEMA))), columns=SCHEMA)
    with pytest.raises(ValueError, match="Invalid ID"):
        validate(frame)
    frame["ID"] = [1, 2]
    frame["default"] = [0, 2]
    with pytest.raises(ValueError, match="Invalid ID or target"):
        validate(frame)

def test_sql_reconciliation_and_orphans(tmp_path):
    from credit_risk.pipeline import reconcile
    from pathlib import Path
    import shutil
    (tmp_path / "artifacts").mkdir()
    shutil.copytree(Path(__file__).parents[1] / "sql", tmp_path / "sql")
    df = pd.DataFrame({"ID": [1, 2], "default": [0, 1], "LIMIT_BAL": [50000, 400000]})
    p = pd.DataFrame({"ID": [1, 2], "model": ["m", "m"], "probability": [.2, .8]})
    sql, _ = reconcile(tmp_path, df, p)
    assert sql.brier.iloc[0] == pytest.approx(.04)
    p.loc[0, "ID"] = 999
    with pytest.raises(ValueError, match="Unmatched"):
        reconcile(tmp_path, df, p)
