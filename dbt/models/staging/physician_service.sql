-- One row per provider-year x HCPCS code x place of service.
SELECT
    CAST(year AS INTEGER)                        AS year,
    Rndrng_NPI                                   AS npi,
    Rndrng_Prvdr_Type                            AS specialty,
    HCPCS_Cd                                     AS hcpcs_cd,
    Place_Of_Srvc                                AS place_of_srvc,
    {{ to_num('Tot_Benes') }}                    AS tot_benes,
    {{ to_num('Tot_Srvcs') }}                    AS tot_srvcs,
    {{ to_num('Tot_Bene_Day_Srvcs') }}           AS bene_day_srvcs,   -- distinct patient-days billed
    {{ to_num('Avg_Mdcr_Pymt_Amt') }}            AS avg_pymt,
    {{ to_num('Avg_Mdcr_Stdzd_Amt') }}           AS avg_stdzd_pymt
FROM {{ source('raw', 'physician_service') }}
