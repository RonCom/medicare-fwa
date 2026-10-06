"""Download CMS provider data (filtered to the configured specialties) and the OIG LEIE.

The service-level file is ~10M rows per year, so we page through the data.cms.gov JSON API with a
provider-type filter and download only the configured specialties.
Output: data/raw/<dataset>/year=<YYYY>/<specialty>.parquet  (all columns as strings).
"""
from __future__ import annotations

import html
import json
import logging
import re
import time

import pandas as pd
import requests

from .config import Config, slug

log = logging.getLogger(__name__)
PAGE = 5000
UA = {"User-Agent": "medicare-fwa-portfolio/0.1 (research; contact via GitHub)"}


def _get(session: requests.Session, url: str, params=None, tries: int = 5):
    for i in range(tries):
        try:
            r = session.get(url, params=params, timeout=180)
            if r.status_code == 200:
                return r
            log.warning("HTTP %s for %s (try %d)", r.status_code, url, i + 1)
        except requests.RequestException as e:
            log.warning("%s for %s (try %d)", e, url, i + 1)
        time.sleep(2 ** i)
    raise RuntimeError(f"Failed after {tries} tries: {url} {params}")


def _norm(title: str) -> str:
    """Lower-case, unescape, collapse spaces, and drop a trailing ' : <date>' (per-year catalog entries)."""
    t = re.sub(r"\s+", " ", html.unescape(title or "")).strip().lower()
    return re.sub(r"\s*:\s*\d{4}(-\d{2}-\d{2})?$", "", t)


def _temporal_year(t) -> int | None:
    """temporal can be a string ('2023-01-01/2023-12-31'), a dict, or a list of PeriodOfTime dicts."""
    if isinstance(t, list):
        t = t[0] if t else None
    if isinstance(t, dict):
        t = t.get("startDate") or t.get("endDate")
    m = re.search(r"(19|20)\d\d", t) if isinstance(t, str) else None
    return int(m.group(0)) if m else None


def api_endpoints_by_year(catalog: list[dict], title: str) -> dict[int, str]:
    """Return {year: api_url} for a dataset title in the data.cms.gov data.json catalog.

    data.cms.gov lists one catalog entry per data year, titled '<title> : YYYY-MM-DD', with the
    year in `temporal` (a list of {startDate, endDate}). The newest entry has two API distributions:
    a 'latest' alias and the year-specific one; we take the year-specific one.
    """
    want = _norm(title)
    matches = [d for d in catalog if _norm(d.get("title", "")) == want]
    if not matches:
        close = sorted({d.get("title") for d in catalog if want.split(" - ")[0] in d.get("title", "").lower()})
        raise KeyError(f"Dataset '{title}' not in catalog. Similar titles: {close[:15]}")
    out: dict[int, str] = {}
    for ds in matches:
        year = _temporal_year(ds.get("temporal")) or _temporal_year(ds.get("title", "").rpartition(":")[2])
        apis = [x for x in ds.get("distribution", []) if "/data-api/v1/dataset/" in (x.get("accessURL") or "")]
        apis.sort(key=lambda x: (x.get("description") or "").lower() == "latest")   # year-specific first
        if year is None or not apis:
            continue
        url = apis[0]["accessURL"].rstrip("/")
        out.setdefault(year, url if url.endswith("/data") else url + "/data")
    if not out:
        raise KeyError(f"'{title}': {len(matches)} catalog entries found but none with a data-api URL")
    return dict(sorted(out.items()))


def load_catalog(cfg: Config, session: requests.Session, force: bool) -> list[dict]:
    """Fetch data.json once and cache it (it's large); --force refreshes it."""
    path = cfg.raw_dir / "cms_catalog.json"
    if force or not path.exists():
        log.info("Reading CMS catalog %s", cfg["cms_catalog_url"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(session, cfg["cms_catalog_url"]).content)
    return json.loads(path.read_text(encoding="utf-8"))["dataset"]


def resolve_years(cfg: Config, available: dict[int, str]) -> list[int]:
    if cfg["years"] == "auto":
        return sorted(available)[-int(cfg["n_years"]):]
    return [y for y in cfg["years"] if y in available]


def fetch_filtered(session, url: str, col: str, value: str) -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        r = _get(session, url, {f"filter[{col}]": value, "size": PAGE, "offset": offset})
        batch = r.json()
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        offset += PAGE
    return pd.DataFrame(rows, dtype="string")


def download_cms(cfg: Config, force: bool = False) -> None:
    s = requests.Session()
    s.headers.update(UA)
    catalog = load_catalog(cfg, s, force)
    for name, spec in cfg["datasets"].items():
        endpoints = api_endpoints_by_year(catalog, spec["title"])
        years = resolve_years(cfg, endpoints)
        log.info("%s: available %s -> using %s", name, list(endpoints), years)
        for y in years:
            log.info("  %s %s -> %s", name, y, endpoints[y])
        for year in years:
            for specialty in cfg["specialties"]:
                dest = cfg.raw_dir / name / f"year={year}" / f"{slug(specialty)}.parquet"
                if dest.exists() and not force:
                    continue
                df = fetch_filtered(s, endpoints[year], spec["type_col"], specialty)
                for alias in (cfg.raw.get("specialty_aliases") or {}).get(specialty, []) if df.empty else []:
                    df = fetch_filtered(s, endpoints[year], spec["type_col"], alias)
                    if not df.empty:
                        log.info("%s %s: '%s' found as '%s'", name, year, specialty, alias)
                        df[spec["type_col"]] = specialty
                        break
                if df.empty:
                    log.warning("0 rows: %s %s '%s' (check spelling of the provider type)", name, year, specialty)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(dest, index=False)
                log.info("%s %s %-40s %8d rows", name, year, specialty, len(df))


def download_leie(cfg: Config, force: bool = False) -> None:
    dest = cfg.raw_dir / "leie" / "UPDATED.csv"
    if dest.exists() and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = _get(requests.Session(), cfg["leie_url"])
    dest.write_bytes(r.content)
    log.info("LEIE saved (%d bytes)", len(r.content))


def download_mue(cfg: Config, years: list[int], force: bool = False) -> None:
    """NCCI practitioner MUE tables, one per data year (Q4 release). Missing years fall back to the
    latest table; the loader then maps each data year to the nearest table it has."""
    s = requests.Session()
    s.headers.update(UA)
    urls = {int(k): v for k, v in (cfg.raw.get("mue_urls") or {}).items()}
    targets = {y: urls.get(y) for y in years}
    targets["latest"] = cfg.raw.get("mue_latest_url")
    for key, url in targets.items():
        if not url:
            continue
        dest = cfg.raw_dir / "mue" / f"{key}.zip"
        if dest.exists() and not force:
            continue
        try:
            r = s.get(url, timeout=180)
        except requests.RequestException as e:
            log.warning("MUE %s: %s", key, e)
            continue
        if r.status_code != 200 or not r.content.startswith(b"PK"):
            log.warning("MUE %s: not found at %s (HTTP %s)", key, url, r.status_code)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        log.info("MUE %s saved (%d bytes)", key, len(r.content))


def download_leie_history(cfg: Config, force: bool = False) -> None:
    """Yearly Internet Archive snapshots of UPDATED.csv + OIG monthly supplements (last 12 months).

    Reinstated providers disappear from the current LEIE, which turns some truly excluded providers
    into negatives. Johnson & Khoshgoftaar (2023) rebuilt the history from archived copies; we do the
    same, taking the first 200-status capture in each January from the Wayback CDX index.
    """
    s = requests.Session()
    s.headers.update(UA)
    hist = cfg.raw_dir / "leie" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    # The file has lived under slightly different URLs; ask the CDX index for each spelling and take
    # the earliest good capture in each calendar year.
    variants = ["oig.hhs.gov/exclusions/downloadables/UPDATED.csv",
                "www.oig.hhs.gov/exclusions/downloadables/UPDATED.csv",
                "oig.hhs.gov/exclusions/downloadables/updated.csv"]
    captures: dict[int, tuple[str, str]] = {}
    for v in variants:
        try:
            r = s.get(cfg["leie_wayback_cdx"], params={"url": v, "output": "json", "filter": "statuscode:200",
                      "fl": "timestamp,original", "from": str(cfg.raw.get("leie_history_from", 2020))}, timeout=180)
        except requests.RequestException as e:
            log.warning("LEIE history: CDX lookup for %s failed (%s)", v, e)
            continue
        if r.status_code != 200:
            log.warning("LEIE history: CDX lookup for %s returned HTTP %s", v, r.status_code)
            continue
        try:
            rows = r.json()[1:] if r.text.strip() else []
        except ValueError:
            rows = []
        log.info("LEIE history: %d captures of %s", len(rows), v)
        for ts, original in rows:
            y = int(ts[:4])
            if y not in captures or ts < captures[y][0]:
                captures[y] = (ts, original)
    for year, (ts, original) in sorted(captures.items()):
        if list(hist.glob(f"UPDATED_{year}*.csv")) and not force:
            continue
        try:
            f = s.get(f"https://web.archive.org/web/{ts}id_/{original}", timeout=300)
        except requests.RequestException as e:
            log.warning("LEIE history %s: %s", year, e)
            continue
        if f.status_code != 200 or b"LASTNAME" not in f.content[:500]:
            log.warning("LEIE history %s: snapshot %s not a LEIE CSV (HTTP %s)", year, ts, f.status_code)
            continue
        (hist / f"UPDATED_{ts}.csv").write_bytes(f.content)
        log.info("LEIE history %s: snapshot %s (%d bytes)", year, ts, len(f.content))
    if not captures:
        log.warning("LEIE history: no archived snapshots found; only OIG's last 12 months of supplements are used")
    today = pd.Timestamp.today()
    # monthly supplements: OIG keeps the previous 12 months
    for k in range(1, 14):
        m = today - pd.DateOffset(months=k)
        for kind in ("excl", "rein"):
            dest = hist / f"{m:%y%m}{kind}.csv"
            if dest.exists() and not force:
                continue
            url = cfg["leie_supplement_pattern"].format(yyyy=f"{m:%Y}", yymm=f"{m:%y%m}", kind=kind)
            try:
                f = s.get(url, timeout=120)
            except requests.RequestException:
                continue
            if f.status_code == 200 and b"LASTNAME" in f.content[:500]:
                dest.write_bytes(f.content)
    n = len(list(hist.glob("*.csv")))
    log.info("LEIE history files: %d", n)


def run(cfg: Config, force: bool = False) -> None:
    download_leie(cfg, force)
    download_cms(cfg, force)
    years = sorted({int(p.name.split("=")[1]) for p in (cfg.raw_dir / "physician_service").glob("year=*")})
    download_mue(cfg, years, force)
    if cfg["validation"].get("leie_source", "current") == "cumulative":
        download_leie_history(cfg, force)
