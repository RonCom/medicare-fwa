-- Provider-year feature mart: one row per (year, npi). Peer comparison happens in Python (fwa/score.py).
SELECT
    p.year, p.npi, p.specialty, p.state, p.last_name, p.first_name,
    p.tot_benes, p.tot_srvcs, p.tot_stdzd_pymt, p.bene_risk_score,
    p.tot_srvcs / NULLIF(p.tot_benes, 0)                                          AS srvcs_per_bene,
    p.tot_stdzd_pymt / NULLIF(p.tot_benes, 0)                                     AS pymt_per_bene,
    p.tot_stdzd_pymt / NULLIF(p.tot_benes, 0) / NULLIF(p.bene_risk_score, 0)      AS pymt_per_bene_risk_adj,
    p.tot_hcpcs_cds                                                               AS distinct_codes,
    e.em_high_share, e.em_lvl5_share, e.em_visits,
    ci.code_intensity_index, ci.max_code_intensity, ci.max_code_intensity_cd,
    bp.timed_units_per_code_day, bp.timed_code_days,
    bp.max_code_days / NULLIF(p.tot_benes, 0)                                     AS service_days_per_bene,
    bp.code_concentration_hhi, bp.passive_modality_share,
    mc.mue_max_ratio, mc.mue_max_cd, mc.codes_over_mue, mc.mue_code_days,
    d.tot_clms                                                                    AS rx_claims,
    d.opioid_tot_clms                                                             AS opioid_claims,
    d.opioid_tot_clms / NULLIF(d.tot_clms, 0)                                     AS opioid_claim_share,
    d.opioid_la_tot_clms / NULLIF(d.opioid_tot_clms, 0)                           AS opioid_la_share,
    d.brnd_tot_clms / NULLIF(d.tot_clms, 0)                                       AS brand_claim_share,
    d.tot_drug_cst / NULLIF(d.tot_benes, 0)                                       AS drug_cost_per_bene,
    d.tot_day_suply / NULLIF(d.tot_clms, 0)                                       AS days_supply_per_claim
FROM {{ ref('physician_provider') }} p
LEFT JOIN {{ ref('em_mix') }}          e  ON e.year = p.year  AND e.npi = p.npi
LEFT JOIN {{ ref('code_intensity') }}  ci ON ci.year = p.year AND ci.npi = p.npi
LEFT JOIN {{ ref('billing_pattern') }} bp ON bp.year = p.year AND bp.npi = p.npi
LEFT JOIN {{ ref('mue_check') }}       mc ON mc.year = p.year AND mc.npi = p.npi
LEFT JOIN {{ ref('partd_provider') }}  d  ON d.year = p.year  AND d.npi = p.npi
WHERE p.tot_benes >= 11
