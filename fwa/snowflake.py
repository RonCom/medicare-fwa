"""Snowflake target: key-pair setup, raw load, dbt build, scoring and reconciliation against DuckDB.

Flow:  DuckDB raw.*  --parquet-->  @RAW.FWA_STAGE  --COPY INTO-->  RAW.*  --dbt-->  STG.* / MART.*
       MART.PROVIDER_FEATURES  --pandas-->  fwa.score  --write_pandas-->  MART.PROVIDER_SCORES
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .load import connect as duck

log = logging.getLogger(__name__)
RAW_TABLES = ["physician_provider", "physician_service", "partd_provider", "leie_all", "mue"]


def _sf(cfg: Config) -> dict:
    s = dict(cfg.raw.get("snowflake") or {})
    s["account"] = os.environ.get("SNOWFLAKE_ACCOUNT") or s.get("account")
    if not s["account"]:
        raise RuntimeError("Snowflake account not set: add snowflake.account to config.local.yaml or set SNOWFLAKE_ACCOUNT")
    s["private_key_path"] = str(Path(os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH", s.get("private_key_path",
                                "~/.snowflake/fwa_rsa_key.p8"))).expanduser())
    return s


def keygen(cfg: Config, force: bool = False) -> None:
    """Create an RSA key pair for the service user; print the public key for snowflake/setup.sql."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    path = Path(_sf(cfg)["private_key_path"])
    if path.exists() and not force:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        log.info("key already exists at %s (use --force to replace)", path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        log.info("private key written to %s (keep it out of the repo)", path)
    pub = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    body = "".join(line for line in pub.splitlines() if "-----" not in line)
    print("\nPaste this into snowflake/setup.sql in place of <PUBLIC_KEY>:\n\n" + body + "\n")


def connect(cfg: Config, schema: str = "RAW"):
    import snowflake.connector
    s = _sf(cfg)
    return snowflake.connector.connect(account=s["account"], user=s["user"], role=s["role"],
                                       warehouse=s["warehouse"], database=s["database"], schema=schema,
                                       authenticator="SNOWFLAKE_JWT", private_key_file=s["private_key_path"])


def load_raw(cfg: Config) -> None:
    """Export DuckDB raw tables to parquet (upper-case column names, so unquoted Snowflake SQL matches)
    and load them with PUT + COPY INTO; table shapes come from INFER_SCHEMA."""
    out = cfg.data_dir / "snowflake_export"
    out.mkdir(parents=True, exist_ok=True)
    con = duck(cfg)
    have = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'raw'").fetchall()}
    missing = [t for t in RAW_TABLES if t not in have]
    if missing:
        con.close()
        raise RuntimeError(f"DuckDB is missing raw.{', raw.'.join(missing)}; run `uv run fwa load` first "
                           "(databases built before the dbt switch kept the MUE table in stg)")
    for t in RAW_TABLES:
        cols = [r[0] for r in con.execute(f"DESCRIBE raw.{t}").fetchall()]
        sel = ", ".join(f'"{c}" AS "{c.upper()}"' for c in cols)
        dest = out / f"{t}.parquet"
        con.execute(f"COPY (SELECT {sel} FROM raw.{t}) TO '{dest.as_posix()}' (FORMAT parquet, ROW_GROUP_SIZE 250000)")
        log.info("exported raw.%s (%d rows)", t, con.execute(f"SELECT count(*) FROM raw.{t}").fetchone()[0])
    con.close()

    sf = connect(cfg)
    cur = sf.cursor()
    cur.execute("CREATE FILE FORMAT IF NOT EXISTS RAW.PARQUET_FMT TYPE = PARQUET")
    cur.execute("CREATE STAGE IF NOT EXISTS RAW.FWA_STAGE FILE_FORMAT = RAW.PARQUET_FMT")
    for t in RAW_TABLES:
        f = (out / f"{t}.parquet").resolve().as_posix()
        cur.execute(f"REMOVE @RAW.FWA_STAGE/{t}/")
        cur.execute(f"PUT 'file://{f}' @RAW.FWA_STAGE/{t}/ AUTO_COMPRESS = FALSE OVERWRITE = TRUE PARALLEL = 8")
        cur.execute(f"""CREATE OR REPLACE TABLE RAW.{t.upper()} USING TEMPLATE (
                          SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*)) FROM TABLE(INFER_SCHEMA(
                            LOCATION => '@RAW.FWA_STAGE/{t}/', FILE_FORMAT => 'RAW.PARQUET_FMT')))""")
        cur.execute(f"""COPY INTO RAW.{t.upper()} FROM @RAW.FWA_STAGE/{t}/
                        FILE_FORMAT = (FORMAT_NAME = 'RAW.PARQUET_FMT') MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE""")
        n = cur.execute(f"SELECT COUNT(*) FROM RAW.{t.upper()}").fetchone()[0]
        log.info("Snowflake RAW.%s: %d rows", t.upper(), n)
    cur.close()
    sf.close()


def features(cfg: Config) -> pd.DataFrame:
    sf = connect(cfg, "MART")
    cur = sf.cursor()
    cur.execute("SELECT * FROM MART.PROVIDER_FEATURES")
    df = cur.fetch_pandas_all()
    sf.close()
    df.columns = [c.lower() for c in df.columns]
    for c in df.columns:                      # Snowflake NUMBER arrives as object/Decimal in some cases
        if df[c].dtype == object and c not in ("npi", "specialty", "state", "last_name", "first_name",
                                                "max_code_intensity_cd", "mue_max_cd"):
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
    return df


def score(cfg: Config) -> pd.DataFrame:
    """Score the Snowflake features with the same Python scorer and write MART.PROVIDER_SCORES back."""
    from snowflake.connector.pandas_tools import write_pandas

    from .score import score_frame
    scores = score_frame(features(cfg), cfg["scoring"])
    out = scores.copy()
    out.columns = [c.upper() for c in out.columns]
    sf = connect(cfg, "MART")
    ok, _, n, _ = write_pandas(sf, out, "PROVIDER_SCORES", schema="MART", auto_create_table=True, overwrite=True,
                               quote_identifiers=False)
    sf.close()
    log.info("Snowflake MART.PROVIDER_SCORES: %d rows written (%s)", n, "ok" if ok else "FAILED")
    return scores


def reconcile(cfg: Config) -> bool:
    """Compare MART.PROVIDER_FEATURES in Snowflake with DuckDB: rows by specialty-year, then every column."""
    sfd = features(cfg).sort_values(["year", "npi"]).reset_index(drop=True)
    con = duck(cfg)
    dd = con.execute("SELECT * FROM mart.provider_features").df().sort_values(["year", "npi"]).reset_index(drop=True)
    con.close()
    lines = ["# Reconciliation: Snowflake vs DuckDB (mart.provider_features)", "",
             f"Rows: DuckDB {len(dd):,} · Snowflake {len(sfd):,}", ""]
    ok = len(dd) == len(sfd)
    counts = dd.groupby(["year", "specialty"]).size().rename("duckdb").to_frame().join(
        sfd.groupby(["year", "specialty"]).size().rename("snowflake"), how="outer")
    ok &= bool((counts["duckdb"] == counts["snowflake"]).all())
    m = dd.merge(sfd, on=["year", "npi"], how="outer", suffixes=("_d", "_s"), indicator=True)
    only = (m["_merge"] != "both").sum()
    ok &= only == 0
    lines += [f"Provider-years in only one system: {only}", "", "| Column | Mismatched rows |", "|---|---|"]
    for c in dd.columns:
        if c in ("year", "npi") or f"{c}_s" not in m:
            continue
        a, b = m[f"{c}_d"], m[f"{c}_s"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a, b = a.astype(float), b.astype(float)
            bad = ~(np.isclose(a, b, rtol=1e-9, atol=1e-9) | (a.isna() & b.isna()))
        else:
            bad = ~((a.astype(str) == b.astype(str)) | (a.isna() & b.isna()))
        n = int(bad[m["_merge"] == "both"].sum())
        ok &= n == 0
        lines.append(f"| {c} | {n:,} |")
    lines += ["", f"**Result: {'MATCH' if ok else 'DIFFERENCES FOUND'}**"]
    (cfg.reports_dir / "reconciliation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("reconciliation: %s (%s)", "MATCH" if ok else "DIFFERENCES FOUND", cfg.reports_dir / "reconciliation.md")
    return ok
