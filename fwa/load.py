"""Load raw parquet/CSV into DuckDB (schema raw); transform() builds staging and marts with dbt (dbt/)."""
from __future__ import annotations

import logging

import duckdb

from .config import ROOT, Config
from .mue import load_mue

log = logging.getLogger(__name__)

# Columns each model needs; checked after load so schema drift fails loudly with a clear message.
REQUIRED = {
    "physician_provider": ["Rndrng_NPI", "Rndrng_Prvdr_Last_Org_Name", "Rndrng_Prvdr_First_Name",
                           "Rndrng_Prvdr_State_Abrvtn", "Rndrng_Prvdr_Type", "Tot_HCPCS_Cds", "Tot_Benes",
                           "Tot_Srvcs", "Tot_Mdcr_Pymt_Amt", "Tot_Mdcr_Stdzd_Amt", "Bene_Avg_Risk_Scre"],
    "physician_service": ["Rndrng_NPI", "Rndrng_Prvdr_Type", "HCPCS_Cd", "Place_Of_Srvc", "Tot_Benes",
                          "Tot_Srvcs", "Tot_Bene_Day_Srvcs", "Avg_Mdcr_Pymt_Amt", "Avg_Mdcr_Stdzd_Amt"],
    "partd_provider": ["Prscrbr_NPI", "Prscrbr_Type", "Tot_Clms", "Tot_Benes", "Tot_Drug_Cst",
                       "Tot_Day_Suply", "Brnd_Tot_Clms", "Opioid_Tot_Clms", "Opioid_LA_Tot_Clms"],
    "leie": ["LASTNAME", "FIRSTNAME", "BUSNAME", "NPI", "STATE", "EXCLTYPE", "EXCLDATE", "REINDATE"],
}


def connect(cfg: Config) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(cfg.db_path))


def load_raw(cfg: Config) -> None:
    con = connect(cfg)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    for name in cfg["datasets"]:
        pattern = (cfg.raw_dir / name / "*" / "*.parquet").as_posix()
        con.execute(f"""
            CREATE OR REPLACE TABLE raw.{name} AS
            SELECT * FROM read_parquet('{pattern}', hive_partitioning = true, union_by_name = true)""")
        n = con.execute(f"SELECT count(*) FROM raw.{name}").fetchone()[0]
        log.info("raw.%s: %d rows", name, n)
    leie = (cfg.raw_dir / "leie" / "UPDATED.csv").as_posix()
    con.execute(f"CREATE OR REPLACE TABLE raw.leie AS SELECT * FROM read_csv('{leie}', all_varchar = true, header = true)")
    log.info("raw.leie: %d rows", con.execute("SELECT count(*) FROM raw.leie").fetchone()[0])
    _load_leie_history(cfg, con)
    _check_columns(con)
    years = [r[0] for r in con.execute("SELECT DISTINCT CAST(year AS INTEGER) FROM raw.physician_service").fetchall()]
    mue = load_mue(cfg.raw_dir, sorted(years))
    con.register("mue_df", mue)
    con.execute("""CREATE OR REPLACE TABLE raw.mue AS
                   SELECT CAST(year AS INTEGER) AS year, CAST(hcpcs_cd AS VARCHAR) AS hcpcs_cd,
                          CAST(mue AS DOUBLE) AS mue, CAST(mai AS INTEGER) AS mai,
                          CAST(rationale AS VARCHAR) AS rationale, CAST(mue_year AS INTEGER) AS mue_year
                   FROM mue_df""")
    log.info("raw.mue: %d rows", con.execute("SELECT count(*) FROM raw.mue").fetchone()[0])
    con.close()


def _load_leie_history(cfg: Config, con) -> None:
    """raw.leie_all = current LEIE, plus archived snapshots and monthly supplements when
    validation.leie_source is 'cumulative'. Duplicates are removed in staging."""
    hist = sorted((cfg.raw_dir / "leie" / "history").glob("*.csv"))
    use_hist = cfg["validation"].get("leie_source", "current") == "cumulative" and hist
    parts = ["SELECT *, 'current' AS leie_file FROM raw.leie"]
    if use_hist:
        files = ", ".join(f"'{p.as_posix()}'" for p in hist)
        con.execute(f"""CREATE OR REPLACE TABLE raw.leie_history AS
            SELECT * REPLACE (regexp_extract(filename, '[^/\\\\]+$') AS filename)
            FROM read_csv([{files}], all_varchar = true, header = true, union_by_name = true, filename = true)""")
        parts.append("SELECT * EXCLUDE (filename), filename AS leie_file FROM raw.leie_history")
    elif cfg["validation"].get("leie_source") == "cumulative":
        log.warning("leie_source is cumulative but no history files found; using the current LEIE only")
    con.execute("CREATE OR REPLACE TABLE raw.leie_all AS " + " UNION ALL BY NAME ".join(parts))
    n_files = len(hist) if use_hist else 0
    log.info("raw.leie_all: %d rows from current + %d history files",
             con.execute("SELECT count(*) FROM raw.leie_all").fetchone()[0], n_files)


def _check_columns(con) -> None:
    problems = []
    for table, cols in REQUIRED.items():
        have = {r[0] for r in con.execute(f"DESCRIBE raw.{table}").fetchall()}
        missing = [c for c in cols if c not in have]
        if missing:
            problems.append(f"raw.{table} missing {missing}; has {sorted(have)}")
    if problems:
        raise ValueError("Source schema changed:\n" + "\n".join(problems))


def transform(cfg: Config, target: str = "duckdb") -> None:
    """Build staging and mart tables with dbt (dbt/ project). target: duckdb (local) or snowflake."""
    from .dbt import run_dbt
    run_dbt(cfg, target, ["build"])
    if target == "duckdb":
        con = connect(cfg)
        log.info("mart.provider_features: %d rows", con.execute("SELECT count(*) FROM mart.provider_features").fetchone()[0])
        con.close()
