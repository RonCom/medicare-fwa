-- E/M level mix (established-patient office visits 99212-99215).
SELECT year, npi,
       SUM(CASE WHEN hcpcs_cd IN ('99214','99215') THEN tot_srvcs END)
         / NULLIF(SUM(CASE WHEN hcpcs_cd IN ('99212','99213','99214','99215') THEN tot_srvcs END), 0) AS em_high_share,
       SUM(CASE WHEN hcpcs_cd = '99215' THEN tot_srvcs END)
         / NULLIF(SUM(CASE WHEN hcpcs_cd IN ('99212','99213','99214','99215') THEN tot_srvcs END), 0) AS em_lvl5_share,
       SUM(CASE WHEN hcpcs_cd IN ('99212','99213','99214','99215') THEN tot_srvcs END)             AS em_visits
FROM {{ ref('physician_service') }}
GROUP BY year, npi
