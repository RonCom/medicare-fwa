-- One row per prescriber-year (Part D).
SELECT
    CAST(year AS INTEGER)                        AS year,
    Prscrbr_NPI                                  AS npi,
    Prscrbr_Type                                 AS specialty,
    {{ to_num('Tot_Clms') }}                     AS tot_clms,
    {{ to_num('Tot_Benes') }}                    AS tot_benes,
    {{ to_num('Tot_Drug_Cst') }}                 AS tot_drug_cst,
    {{ to_num('Tot_Day_Suply') }}                AS tot_day_suply,
    {{ to_num('Brnd_Tot_Clms') }}                AS brnd_tot_clms,
    {{ to_num('Opioid_Tot_Clms') }}              AS opioid_tot_clms,
    {{ to_num('Opioid_LA_Tot_Clms') }}           AS opioid_la_tot_clms
FROM {{ source('raw', 'partd_provider') }}
