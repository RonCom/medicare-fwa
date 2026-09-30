-- Billing-pattern metrics (built for physical therapy, computed for every specialty):
--  timed units per code-day (8-minute rule), service days per beneficiary, payment concentration
--  (Herfindahl index across codes) and the share of passive / unattended modalities.
WITH s AS (
    SELECT year, npi, hcpcs_cd,
           SUM(tot_srvcs) AS srvcs, SUM(bene_day_srvcs) AS days, SUM(tot_srvcs * avg_stdzd_pymt) AS pymt,
           CASE WHEN hcpcs_cd IN ('97032','97033','97034','97035','97036','97110','97112','97113','97116','97124',
                                  '97140','97530','97533','97535','97537','97542','97750','97755','97760','97761',
                                  '97763') THEN 1 ELSE 0 END AS is_timed,
           CASE WHEN hcpcs_cd IN ('G0283','97010','97012','97014','97016','97018','97022','97024','97026','97028')
                THEN 1 ELSE 0 END AS is_passive
    FROM {{ ref('physician_service') }}
    GROUP BY year, npi, hcpcs_cd
), t AS (
    SELECT year, npi, SUM(pymt) AS tot_pymt, SUM(srvcs) AS tot_srvcs FROM s GROUP BY year, npi
)
SELECT s.year, s.npi,
       SUM(CASE WHEN is_timed = 1 THEN srvcs END) / NULLIF(SUM(CASE WHEN is_timed = 1 THEN days END), 0) AS timed_units_per_code_day,
       SUM(CASE WHEN is_timed = 1 THEN days END)                                                   AS timed_code_days,
       MAX(days)                                                                                   AS max_code_days,
       SUM(POWER(pymt / NULLIF(t.tot_pymt, 0), 2))                                                 AS code_concentration_hhi,
       COALESCE(SUM(CASE WHEN is_passive = 1 THEN srvcs END), 0) / NULLIF(MAX(t.tot_srvcs), 0)     AS passive_modality_share
FROM s JOIN t ON t.year = s.year AND t.npi = s.npi
GROUP BY s.year, s.npi
