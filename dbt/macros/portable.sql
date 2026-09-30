{# Small cross-database helpers: the same models run on DuckDB and Snowflake. #}

{# 'YYYYMMDD' text -> DATE, NULL when blank, 00000000 or invalid #}
{% macro try_yyyymmdd(col) -%}
  {%- if target.type == 'snowflake' -%}
    TRY_TO_DATE(NULLIF(TRIM({{ col }}), '00000000'), 'YYYYMMDD')
  {%- else -%}
    TRY_STRPTIME(NULLIF(TRIM({{ col }}), '00000000'), '%Y%m%d')::DATE
  {%- endif -%}
{%- endmacro %}

{# numeric text -> DOUBLE, NULL when blank/suppressed #}
{% macro to_num(col) -%}
  TRY_CAST({{ col }} AS DOUBLE)
{%- endmacro %}
