# Explain the project honestly

**30-second overview:** I built a reproducible credit-risk validation study using a public 30,000-client dataset. I compared logistic regression with calibrated gradient boosting, prevented identical predictor profiles from crossing partitions, and reconciled SQL reporting against Python. On 6,010 held-out records the challenger reached 0.780 ROC-AUC and 0.557 average precision. I also implemented PSI monitoring with explicitly synthetic shifts and documented why this historical benchmark is not production-ready.

## Questions to be ready for

- Why not accuracy? Default prevalence is about 22%; ranking and calibrated probabilities provide more useful diagnostics than majority-class accuracy.
- Why AP and AUC? AUC summarizes ranking over thresholds; AP focuses on positive retrieval and depends on prevalence. Neither defines a lending policy.
- Why group duplicates? Identical profiles could otherwise leak across random partitions and overstate generalization.
- Why calibration? Good ranking need not mean reliable probabilities. Sigmoid calibration uses training folds only; test Brier and calibration plots assess the result.
- What is the biggest weakness? One old cohort, no out-of-time or external test, and no measured operating costs.
- Does PSI prove model failure? No. It detects changes in binned marginal distributions; stable PSI also cannot rule out changing label relationships.
- What is original? The implementation, contracts, reconciliation, report and integration. The algorithms are standard, and source inspirations are attributed.
- Is this a CECL model? No: no lifetime forecast, LGD/EAD estimates, scenario weighting or allowance calculation.

## Resume wording after reproducing and understanding the project

- Built a Python/SQL credit-default modeling pipeline on 30,000 public records; compared logistic regression with calibrated gradient boosting, achieving 0.780 ROC-AUC on 6,010 held-out clients.
- Implemented duplicate-aware validation, ROC-AUC/AP/KS and calibration diagnostics, PSI drift monitoring, and independently reconciled SQL/Python reports with automated tests.

Use only claims you can explain and reproduce. The project was developed with AI assistance; do not claim solo-from-scratch statistical invention, bank deployment, business impact or a production validation approval.
