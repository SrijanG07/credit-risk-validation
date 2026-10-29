"""Checksum-verified public data and a strict feature contract."""
from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
TARGET = "default"
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
# Demographics retained for audit only, never used by the predictive models.
FEATURES = ["LIMIT_BAL"] + PAY + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
SCHEMA = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)] + [TARGET]

def load_data(root: Path) -> pd.DataFrame:
    path = root / "data/raw/uci-default.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=90) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError("Download checksum mismatch; investigate source version before proceeding")
        path.write_bytes(payload)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("Cached dataset checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [n for n in archive.namelist() if n.endswith(".xls")]
        if len(members) != 1:
            raise ValueError("Unexpected archive layout")
        df = pd.read_excel(io.BytesIO(archive.read(members[0])), header=1)
    df = df.rename(columns={"default payment next month": TARGET})
    validate(df)
    return df

def validate(df):
    if list(df.columns) != SCHEMA:
        raise ValueError("Unexpected column schema")
    if df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError("Missing or nonfinite source values")
    if not df.ID.is_unique or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Invalid ID or target")
    if (df.LIMIT_BAL <= 0).any():
        raise ValueError("Invalid credit limits")

def split_data(df, seed=42):
    # Identical predictor rows cannot straddle partitions, even with different labels.
    groups = pd.util.hash_pandas_object(df[FEATURES], index=False).to_numpy()
    folds = np.empty(len(df), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, ix) in enumerate(splitter.split(df, df[TARGET], groups)):
        folds[ix] = fold
    partitions = {"train": np.flatnonzero(folds >= 2), "validation": np.flatnonzero(folds == 1), "test": np.flatnonzero(folds == 0)}
    return partitions, groups
