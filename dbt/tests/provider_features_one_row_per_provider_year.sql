-- Fails if any (year, npi) appears more than once.
SELECT year, npi, COUNT(*) AS n
FROM {{ ref('provider_features') }}
GROUP BY year, npi
HAVING COUNT(*) > 1
