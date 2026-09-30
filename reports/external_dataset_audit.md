# Audit: Kaggle "Healthcare Fraud Detection Dataset" (nudratabbas)

File: data/external/healthcare_fraud_detection.csv · 10,000 claims · 300 providers · 20 columns · fraud rate 8.3%.

Verdict: synthetic, label generated from a few columns; not used for validation or comparison.

- Label documentation: none on the dataset page.
- Approved / claimed amount ratio alone separates the label (AUC 0.99): fraud median 0.54 vs 0.88.
- Days between service and claim: AUC 0.94 (fraud median 3 days vs 15).
- One rule, ratio < 0.77 and days <= 6, flags 929 claims with precision 0.851 and recall 0.954.
- Claim_Status leaks the outcome (fraud rate 2% Approved vs 23% Rejected/Pending).
- Clinical fields carry no signal (single-feature AUC ~0.50) and are drawn independently: e.g. 12% of cardiology
  claims are physical-therapy code 97110, and diagnoses are unrelated to procedures.
- Provider IDs are not NPIs, so the data cannot be linked to CMS or the OIG exclusion list.
