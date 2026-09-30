# Experiments vs frozen baseline

Label: same-NPI OIG exclusion within the horizon after the data year (cumulative LEIE when enabled). Scores ranked within specialty × year. 95% CIs resample providers.

| Run | What | AUC | Top 5% lift | Top 5% recall | Top 5% $ recall |
|---|---|---|---|---|---|
| baseline_risk_score | production score (rules 2 : IF 1) | 0.70 [0.64–0.75] | 3.4 [1.9–4.9] | 17% [9–24%] | 21% [8–37%] |
| baseline_composite_pct | rules only | 0.71 [0.65–0.76] | 3.3 [1.9–4.9] | 16% [9–24%] | 12% [4–26%] |
| baseline_iforest_pct | Isolation Forest only | 0.67 [0.59–0.72] | 3.6 [2.0–5.3] | 18% [10–26%] | 31% [10–56%] |
| A1 | XGBoost, 5-fold CV grouped by NPI | 0.65 [0.58–0.71] | 2.6 [1.4–4.2] | 13% [7–21%] | 8% [2–16%] |
| A2 | XGBoost, NPI-grouped + trained on years <= 2019, tested on later years (test rows only) | 0.67 [0.58–0.75] | 4.0 [1.6–6.7] | 20% [8–34%] | 18% [3–49%] |
| B | rule score on metrics residualized for patient mix (log scale, OLS per specialty-year) | 0.67 [0.61–0.72] | 2.8 [1.4–4.4] | 14% [7–22%] | 21% [6–40%] |
| C_ecod | baseline rules + ECOD in place of Isolation Forest (2:1) | 0.70 [0.63–0.76] | 3.4 [2.0–4.9] | 17% [10–24%] | 21% [8–37%] |
| C_cblof | baseline rules + CBLOF in place of Isolation Forest (2:1) | 0.72 [0.65–0.76] | 3.6 [2.0–5.3] | 18% [10–26%] | 23% [9–41%] |
| C_ecod_alone | ECOD alone | 0.63 [0.55–0.70] | 2.1 [1.1–3.3] | 11% [6–16%] | 13% [3–29%] |
| C_cblof_alone | CBLOF alone | 0.68 [0.62–0.74] | 3.4 [1.9–4.9] | 17% [10–25%] | 27% [10–46%] |

## Replication ladder (Johnson & Khoshgoftaar 2023 setup → this project's)

| Step | Setup | Positive providers | AUC |
|---|---|---|---|
| D1 | their label, all specialties pooled, row-level CV, pooled AUC | 87 | 0.93 [0.91–0.95] |
| D2 | their label, NPI-grouped CV, pooled AUC | 87 | 0.82 [0.77–0.88] |
| D3 | their label, NPI-grouped CV, AUC within specialty-year | 87 | 0.66 [0.60–0.72] |
| D4 | our label (future exclusion), NPI-grouped CV, within specialty-year | 66 | 0.62 [0.55–0.68] |

## Pre-registered test: CBLOF vs Isolation Forest on held-out years 2016-2019

34 later-excluded providers. Paired bootstrap (1,000 provider resamples), CBLOF minus baseline:

- AUC difference: 0.009 [-0.013–0.032]
- Top-5% dollar-recall difference: -0.008 [-0.069–0.042]
- Decision: **keep Isolation Forest**

A1 top features (mean |SHAP|): mue_max_ratio__adj 1.077, timed_units_per_code_day__adj 1.054, cm_risk_score 0.816, cm_female_share 0.587, cm_avg_age 0.469, code_intensity_index__adj 0.452, service_days_per_bene__adj 0.426, cm_chronic_pct 0.401
