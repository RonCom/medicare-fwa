-- NCCI practitioner MUE per data year (parsed from the CMS zips by fwa/mue.py).
SELECT CAST(year AS INTEGER) AS year, CAST(hcpcs_cd AS VARCHAR) AS hcpcs_cd, CAST(mue AS DOUBLE) AS mue,
       CAST(mai AS INTEGER) AS mai, CAST(rationale AS VARCHAR) AS rationale, CAST(mue_year AS INTEGER) AS mue_year
FROM {{ source('raw', 'mue') }}
