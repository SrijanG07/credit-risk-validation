# Model validation memo

## Scope and decision

Estimate next-month default probabilities for existing credit-card accounts represented by the UCI cohort. Choose between a fixed linear baseline and a nonlinear challenger by validation ROC-AUC. The calibrated challenger was selected (validation AUC 0.7737 vs 0.7121), then both candidates were evaluated on the locked test partition.

The challenger improves held-out ranking and probability calibration: AUC 0.7795, AP 0.5574, KS 0.4355, Brier 0.1327, ECE 0.0114. Baseline AUC is 0.7223. These results support further research on this cohort; they do not justify a production lending policy. No operating threshold is recommended without loss/revenue assumptions and appropriate approval.

## Specification

Logistic: median imputation, standard scaling, L2 logistic regression, C=1, max_iter=3000. Continuous treatment of repayment-status codes is deliberately a simple baseline; sentinel categories are a known limitation.

Challenger: scikit-learn HistGradientBoostingClassifier; 180 iterations, learning rate .05, 15 leaves, L2=5, early stopping disabled. Three grouped training folds each fit an estimator and sigmoid calibrator; final predictions average the calibrated members. This is not XGBoost. The fixed configurations are intentionally modest, not claimed optimal.

Validation selection uses ROC-AUC only. Calibration metrics are diagnostic, not additional hidden selection rules. AP is a step-weighted average precision statistic; KS is max(TPR-FPR) across score thresholds; ECE is a 10 equal-width probability-bin weighted absolute gap; calibration plots use 10 equal-count bins.

## Evidence and checks

- Source SHA, schema, unique IDs, binary labels and finite values checked on ingestion.
- 817 repeated predictor rows grouped before splitting. This protects against exact profile overlap, not every form of unobserved dependence.
- Training preprocessing and calibration never fit test or validation labels.
- All records assigned once; no predictor group appears in multiple partitions.
- SQL independently reconciles counts and rates with Python; reporting stops on discrepancies.
- AUC uncertainty uses 300 cluster bootstrap replicates; intervals do not include model-fitting uncertainty.
- Subgroup counts, observed/predicted rates and AUC are exported for sex and age. These are descriptive diagnostics, not a fairness certification. Additional measures, uncertainty and legal/contextual review would be needed for an operational fairness assessment.

## Findings and limitations

The benchmark's age, geography, product coverage and selection of existing customers limit transportability. Removing protected characteristics does not remove correlated proxies. Bill/payment fields can be negative or carry unusual codes; the ingestion does not silently erase these valid or documented source conventions. Repayment codes -2/0 require source-context interpretation and should be investigated before deployment.

No temporal evaluation is possible from the provided cohort. Repayment histories span April–September 2005; splitting rows by ID would invent temporal ordering. No model is claimed to forecast losses, LGD/EAD, macroeconomic stress, CECL allowances or CCAR capital.

Recommended next steps: independent temporal cohort, repeated grouped training evaluations, repayment-code sensitivity analysis, calibrated decision costs, segment confidence intervals, and documentation of data availability at the true scoring point.
