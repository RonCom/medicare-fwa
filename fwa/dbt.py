"""Run the dbt project against DuckDB (local) or Snowflake."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .config import ROOT, Config

log = logging.getLogger(__name__)
DBT_DIR = ROOT / "dbt"


def run_dbt(cfg: Config, target: str, args: list[str]) -> None:
    from dbt.cli.main import dbtRunner

    os.environ["FWA_DB_PATH"] = str(cfg.db_path)
    os.environ["FWA_DBT_TARGET"] = target
    sf = cfg.raw.get("snowflake") or {}
    key = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH") or str(Path(sf.get("private_key_path", "~/.snowflake/fwa_rsa_key.p8")).expanduser())
    os.environ.setdefault("SNOWFLAKE_PRIVATE_KEY_PATH", key)
    for k, env in (("account", "SNOWFLAKE_ACCOUNT"), ("user", "SNOWFLAKE_USER"), ("role", "SNOWFLAKE_ROLE"),
                   ("warehouse", "SNOWFLAKE_WAREHOUSE"), ("database", "SNOWFLAKE_DATABASE")):
        if sf.get(k):
            os.environ.setdefault(env, str(sf[k]))
    cli = args + ["--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR), "--target", target]
    log.info("dbt %s", " ".join(cli))
    res = dbtRunner().invoke(cli)
    if not res.success:
        raise RuntimeError(f"dbt {' '.join(args)} failed on target {target}: {res.exception or 'see dbt log above'}")
