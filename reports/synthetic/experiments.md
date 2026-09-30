# Experiments vs frozen baseline

Label: same-NPI OIG exclusion within the horizon after the data year (cumulative LEIE when enabled). Scores ranked within specialty × year. 95% CIs resample providers.

| Run | What | AUC | Top 5% lift | Top 5% recall | Top 5% $ recall |
|---|---|---|---|---|---|
| baseline_risk_score | production score (rules 2 : IF 1) | 0.76 [0.56–0.94] | 12.2 [7.3–17.3] | 61% [36–89%] | 80% [49–98%] |
| baseline_composite_pct | rules only | 0.76 [0.54–0.95] | 11.6 [6.7–17.2] | 58% [33–86%] | 77% [46–93%] |
| baseline_iforest_pct | Isolation Forest only | 0.80 [0.64–0.94] | 12.9 [8.9–17.5] | 65% [43–89%] | 81% [50–98%] |
| C_ecod | baseline rules + ECOD in place of Isolation Forest (2:1) | 0.76 [0.56–0.94] | 12.2 [7.3–17.2] | 61% [36–89%] | 80% [49–98%] |
| C_cblof | baseline rules + CBLOF in place of Isolation Forest (2:1) | 0.77 [0.57–0.95] | 12.2 [7.4–17.7] | 61% [36–89%] | 80% [49–98%] |
| C_ecod_alone | ECOD alone | 0.80 [0.63–0.92] | 10.9 [7.1–15.2] | 55% [35–80%] | 75% [46–90%] |
| C_cblof_alone | CBLOF alone | 0.81 [0.66–0.95] | 12.2 [7.6–17.5] | 61% [36–89%] | 80% [49–98%] |
