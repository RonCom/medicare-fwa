"""Parse CMS NCCI practitioner MUE tables.

The zip holds a CSV and/or XLSX with several disclaimer lines above the column header, and column
names that have varied slightly across releases. We locate the header row by the word "HCPCS"
and match columns by keywords, so small format changes don't break the load.
"""
from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _rows_from_member(z: zipfile.ZipFile, name: str) -> list[list[str]]:
    data = z.read(name)
    if name.lower().endswith(".csv"):
        text = data.decode("utf-8-sig", errors="replace")
        return [r for r in csv.reader(io.StringIO(text))]
    df = pd.read_excel(io.BytesIO(data), header=None, dtype=str)
    return df.fillna("").astype(str).values.tolist()


def _pick(cols: list[str], *keys: str) -> int | None:
    for i, c in enumerate(cols):
        low = c.lower()
        if all(k in low for k in keys):
            return i
    return None


def parse_mue_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if n.lower().endswith((".csv", ".xlsx", ".xls"))]
        if not members:
            raise ValueError(f"{path.name}: no CSV/XLSX inside ({z.namelist()})")
        members.sort(key=lambda n: (not n.lower().endswith(".csv"), n))       # prefer CSV
        rows = _rows_from_member(z, members[0])
    hdr = next((i for i, r in enumerate(rows) if any("hcpcs" in str(c).lower() for c in r)), None)
    if hdr is None:
        raise ValueError(f"{path.name}: no header row containing 'HCPCS'")
    cols = [str(c).strip() for c in rows[hdr]]
    i_code = _pick(cols, "hcpcs")
    i_mue = _pick(cols, "mue", "value") or _pick(cols, "mue")
    i_mai = _pick(cols, "adjudication")
    i_rat = _pick(cols, "rationale")
    if i_code is None or i_mue is None:
        raise ValueError(f"{path.name}: cannot find code/MUE columns in {cols}")
    out = []
    for r in rows[hdr + 1:]:
        if len(r) <= max(i_code, i_mue):
            continue
        code = str(r[i_code]).strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{5}", code):
            continue
        try:
            mue = float(str(r[i_mue]).strip())
        except ValueError:
            continue
        mai = re.match(r"\s*(\d)", str(r[i_mai])) if i_mai is not None and len(r) > i_mai else None
        out.append({"hcpcs_cd": code, "mue": mue, "mai": int(mai.group(1)) if mai else None,
                    "rationale": str(r[i_rat]).strip() if i_rat is not None and len(r) > i_rat else None})
    df = pd.DataFrame(out).drop_duplicates("hcpcs_cd")
    log.info("MUE %s: %d codes (from %s)", path.stem, len(df), members[0])
    return df


def load_mue(raw_dir: Path, years: list[int]) -> pd.DataFrame:
    """One MUE table per data year; a year without a file borrows the nearest year that has one."""
    files = {int(p.stem): p for p in (raw_dir / "mue").glob("*.zip") if p.stem.isdigit()}
    latest = raw_dir / "mue" / "latest.zip"
    if latest.exists() and not files:
        files = {max(years): latest}
    if not files:
        log.warning("No MUE tables in %s; the MUE metric will be empty", raw_dir / "mue")
        return pd.DataFrame(columns=["year", "hcpcs_cd", "mue", "mai", "rationale", "mue_year"])
    parsed = {y: parse_mue_zip(p) for y, p in files.items()}
    parts = []
    for y in years:
        src = min(parsed, key=lambda k: (abs(k - y), -k))
        if src != y:
            log.warning("MUE for %s not found; using %s", y, src)
        parts.append(parsed[src].assign(year=y, mue_year=src))
    return pd.concat(parts, ignore_index=True)
