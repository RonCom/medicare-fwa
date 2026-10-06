-- OIG exclusions, de-duplicated across the current list, archived snapshots and monthly supplements.
-- NPI is missing (zeros) on ~89% of rows in the current list; dates are YYYYMMDD with 00000000 = none. A reinstatement date is kept
-- when any source has one; in_current_list = still excluded today.
WITH r AS (
    SELECT
        NULLIF(NULLIF(TRIM(NPI), ''), '0000000000')   AS npi,
        UPPER(TRIM(LASTNAME))                         AS last_name,
        UPPER(TRIM(FIRSTNAME))                        AS first_name,
        UPPER(TRIM(BUSNAME))                          AS business_name,
        UPPER(TRIM(STATE))                            AS state,
        LOWER(REPLACE(TRIM(EXCLTYPE), ' ', ''))       AS excl_type,
        {{ try_yyyymmdd('EXCLDATE') }}                AS excl_date,
        {{ try_yyyymmdd('REINDATE') }}                AS rein_date,
        CASE WHEN leie_file = 'current' THEN 1 ELSE 0 END AS is_current
    FROM {{ source('raw', 'leie_all') }}
)
SELECT npi, last_name, first_name, business_name, state, excl_type, excl_date,
       MAX(rein_date)          AS rein_date,
       MAX(is_current) = 1     AS in_current_list,
       COUNT(*)                AS n_sources
FROM r
WHERE excl_date IS NOT NULL
GROUP BY npi, last_name, first_name, business_name, state, excl_type, excl_date
