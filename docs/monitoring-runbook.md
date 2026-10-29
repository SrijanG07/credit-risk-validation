# Monitoring runbook

1. Validate schema, missingness, unique identifiers and row counts before scoring. Quarantine failures; do not silently repair changed source definitions.
2. Compare feature and prediction PSI against a versioned reference. This example uses validation records, never model-training predictions, as the reference score distribution.
3. For PSI >0.10, investigate cohort mix, ingestion changes, coding changes and sample size. For >0.25, prioritize review. These are illustrative investigation thresholds, not automatic rejection/retraining rules.
4. Reconcile SQL and Python aggregates. Check unmatched IDs, duplicate keys and target maturity before distributing a report.
5. When labels mature, recompute AUC, AP, KS, calibration and subgroup diagnostics. Covariate drift alone neither proves nor rules out concept drift or performance deterioration.
6. Record the finding, evidence, impact, owner and remediation. Review a challenger on an untouched evaluation cohort before replacing the model.

The supplied synthetic scenario is deliberately detectable. It shifts LIMIT_BAL and repayment status simultaneously; use individual PSI rows to identify affected fields. Historical labels are not valid outcomes for those counterfactual records, so no synthetic AUC or loss forecast is reported.
