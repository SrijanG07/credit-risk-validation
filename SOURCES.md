# Sources, licenses and originality

Sources inspected on 2026-10-29.

| Source | Verified license | How used |
|---|---|---|
| [ShichenXie/scorecardpy](https://github.com/ShichenXie/scorecardpy) | [MIT, copyright Shichen Xie](https://github.com/ShichenXie/scorecardpy/blob/master/LICENSE) | Conceptual inspiration for credit-scoring evaluation including KS and PSI. No source code, notebooks or assets copied. |
| [evidentlyai/evidently](https://github.com/evidentlyai/evidently) | [Apache-2.0](https://github.com/evidentlyai/evidently/blob/main/LICENSE) | Conceptual inspiration for reference/current monitoring and readable reports. No source code or assets copied. |
| [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Public source data; downloaded on demand, never represented as original collection. |

Dataset citation: Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H.

Source archive: https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip

SHA-256: `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`

Transformations: rename target to `default`; exclude ID and demographics from predictors; assign deterministic grouped partitions; generate aggregated metrics and synthetic feature perturbations. The dataset is a historical Taiwanese credit-card cohort. The source publication is Yeh and Lien (2009), *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*, Expert Systems with Applications.

Runtime libraries retain their own licenses: scikit-learn (BSD-3-Clause), NumPy (BSD-3-Clause), pandas (BSD-3-Clause), Matplotlib (Matplotlib license), xlrd (BSD-style). They are installed as dependencies rather than vendored. Standard logistic regression, gradient boosting, calibration, ROC, AP, KS, Brier and PSI are not claimed as inventions.

The implementation was authored with AI assistance. The owner should reproduce it, review the code, and understand the methodology before describing it as personal hands-on expertise in an interview. No resume, contact information or recruitment document is included in the public repository.
