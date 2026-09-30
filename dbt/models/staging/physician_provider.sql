-- One row per provider-year. Suppressed CMS cells arrive blank -> NULL.
SELECT
    CAST(year AS INTEGER)                        AS year,
    Rndrng_NPI                                   AS npi,
    UPPER(TRIM(Rndrng_Prvdr_Last_Org_Name))      AS last_name,
    UPPER(TRIM(Rndrng_Prvdr_First_Name))         AS first_name,
    Rndrng_Prvdr_State_Abrvtn                    AS state,
    Rndrng_Prvdr_Type                            AS specialty,
    {{ to_num('Tot_HCPCS_Cds') }}                AS tot_hcpcs_cds,
    {{ to_num('Tot_Benes') }}                    AS tot_benes,
    {{ to_num('Tot_Srvcs') }}                    AS tot_srvcs,
    {{ to_num('Tot_Mdcr_Pymt_Amt') }}            AS tot_pymt,
    {{ to_num('Tot_Mdcr_Stdzd_Amt') }}           AS tot_stdzd_pymt,
    {{ to_num('Bene_Avg_Risk_Scre') }}           AS bene_risk_score
FROM {{ source('raw', 'physician_provider') }}
