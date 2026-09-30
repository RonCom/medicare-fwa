-- NCCI MUE check. Units per patient-day per code is an average; above the MUE means some days
-- exceeded the limit. Codes capped at 1 sit at ratio 1 for almost everyone, so the scored ratio
-- uses codes with MUE >= 2; codes_over_mue counts any code's excess.
WITH s AS (
    SELECT year, npi, hcpcs_cd, SUM(tot_srvcs) AS srvcs, SUM(bene_day_srvcs) AS days
    FROM {{ ref('physician_service') }}
    GROUP BY year, npi, hcpcs_cd
    HAVING SUM(bene_day_srvcs) >= 11
), j AS (
    SELECT s.year, s.npi, s.hcpcs_cd, s.days, m.mue, s.srvcs / s.days / m.mue AS ratio
    FROM s JOIN {{ ref('mue') }} m ON m.year = s.year AND m.hcpcs_cd = s.hcpcs_cd
    WHERE m.mue > 0
)
-- top code chosen deterministically (highest ratio among MUE >= 2, then code)
, top_code AS (
    SELECT year, npi, hcpcs_cd,
           ROW_NUMBER() OVER (PARTITION BY year, npi ORDER BY ratio DESC NULLS LAST, hcpcs_cd) AS rn
    FROM j WHERE mue >= 2
)
SELECT j.year, j.npi,
       MAX(CASE WHEN j.mue >= 2 THEN j.ratio END)                 AS mue_max_ratio,
       MAX(t.hcpcs_cd)                                            AS mue_max_cd,
       SUM(CASE WHEN j.ratio > 1 THEN 1 ELSE 0 END)               AS codes_over_mue,
       SUM(j.days)                                                AS mue_code_days
FROM j
LEFT JOIN top_code t ON t.year = j.year AND t.npi = j.npi AND t.rn = 1
GROUP BY j.year, j.npi
