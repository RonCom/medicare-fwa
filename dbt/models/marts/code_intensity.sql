-- For each code, the provider's services per beneficiary vs the specialty median for that code;
-- summarised as a payment-weighted index and the single most extreme code.
WITH per_code AS (
    SELECT year, npi, specialty, hcpcs_cd,
           SUM(tot_srvcs) / NULLIF(MAX(tot_benes), 0) AS spb,
           SUM(tot_srvcs * avg_stdzd_pymt)            AS stdzd_pymt
    FROM {{ ref('physician_service') }}
    WHERE tot_benes >= 11
    GROUP BY year, npi, specialty, hcpcs_cd
), peer AS (
    SELECT year, specialty, hcpcs_cd, MEDIAN(spb) AS peer_spb, COUNT(*) AS n_peer
    FROM per_code GROUP BY year, specialty, hcpcs_cd
), joined AS (
    SELECT c.year, c.npi, c.hcpcs_cd, c.stdzd_pymt, c.spb / NULLIF(p.peer_spb, 0) AS ratio
    FROM per_code c
    JOIN peer p ON p.year = c.year AND p.specialty = c.specialty AND p.hcpcs_cd = c.hcpcs_cd
    WHERE p.n_peer >= 30
)
-- top code chosen deterministically (highest ratio, then code) so DuckDB and Snowflake agree on ties
, top_code AS (
    SELECT year, npi, hcpcs_cd,
           ROW_NUMBER() OVER (PARTITION BY year, npi ORDER BY ratio DESC NULLS LAST, hcpcs_cd) AS rn
    FROM joined
)
SELECT j.year, j.npi,
       SUM(j.ratio * j.stdzd_pymt) / NULLIF(SUM(j.stdzd_pymt), 0) AS code_intensity_index,
       MAX(j.ratio)                                              AS max_code_intensity,
       MAX(t.hcpcs_cd)                                           AS max_code_intensity_cd
FROM joined j
LEFT JOIN top_code t ON t.year = j.year AND t.npi = j.npi AND t.rn = 1
GROUP BY j.year, j.npi
