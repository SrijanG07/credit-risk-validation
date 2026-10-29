"""Discrimination, calibration, uncertainty and reference-fitted PSI."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss, log_loss

def evaluate(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("Expected nonempty one-dimensional labels and probabilities of equal length")
    if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("Evaluation requires binary labels with both classes present")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities")
    fpr, tpr, _ = roc_curve(y, p)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(y[bins == b].mean() - p[bins == b].mean()) for b in np.unique(bins))
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc_ap": float(average_precision_score(y, p)), "ks": float(np.max(tpr - fpr)), "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)), "ece_10": float(ece)}

def bootstrap_auc(y, p, groups, n=300, seed=2026):
    """Cluster bootstrap: identical predictor records are sampled as units."""
    rng = np.random.default_rng(seed)
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    evaluate(y, p)
    if groups.ndim != 1 or len(groups) != len(y) or n < 20:
        raise ValueError("Bootstrap requires aligned groups and at least 20 replicates")
    unique = np.unique(groups)
    lookup = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(n):
        ix = np.concatenate([lookup[g] for g in rng.choice(unique, len(unique), replace=True)])
        if np.unique(y[ix]).size == 2:
            values.append(roc_auc_score(y[ix], p[ix]))
    if len(values) < 20:
        raise ValueError("Insufficient two-class bootstrap samples for an interval")
    return [float(v) for v in np.quantile(values, [.025, .975])]

def psi(reference, current, bins=10):
    """Reference quantiles, infinite tails, separate missing bin, 0.5 count smoothing.

    Constant reference gets a narrow central bin so a moved constant is detectable.
    No time inference or statistical significance is implied by this statistic.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    if not len(reference) or not len(current):
        raise ValueError("PSI requires nonempty samples")
    finite = reference[np.isfinite(reference)]
    if not len(finite):
        edges = np.array([-np.inf, np.inf])
    elif np.unique(finite).size == 1:
        value = finite[0]
        width = max(abs(value), 1) * 1e-8
        edges = np.array([-np.inf, value - width, value + width, np.inf])
    else:
        edges = np.r_[-np.inf, np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1])), np.inf]
    def distribution(a):
        counts = np.r_[np.histogram(a[np.isfinite(a)], bins=edges)[0], (~np.isfinite(a)).sum()].astype(float) + .5
        return counts / counts.sum()
    expected, actual = distribution(reference), distribution(current)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
