"""Do high risk scores precede OIG exclusion?

For a provider scored on data year Y:
  label      = 1 if the LEIE has an exclusion for the same NPI dated after Dec 31 of Y and within
               `horizon_years` (primary, high-confidence label)
  label_name = same, but also counting LEIE rows with no NPI matched on last+first name+state
               (sensitivity check; common names make these noisy)
Providers already excluded by the end of Y are dropped.

Provider-years are not independent (the same provider appears up to 4 times), so confidence
intervals come from a bootstrap that resamples *providers*, not rows.

Caveats: the LEIE download lists current exclusions (reinstated providers drop off), exclusion is a
lagging and incomplete proxy for FWA, and positives are rare, so intervals are wide.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from .config import Config
from .load import connect

log = logging.getLogger(__name__)
# Within-peer-group percentiles only. Raw z's are not comparable across specialties: pain specialties
# have higher raw composite z's AND a ~20x higher exclusion rate, which inflates a pooled AUC.
SCORES = ("risk_score", "composite_pct", "iforest_pct")
TIERS = (("top1", 0.99), ("top5", 0.95))


def labelled(cfg: Config) -> pd.DataFrame:
    v = cfg["validation"]
    con = connect(cfg)
    df = con.execute("""
        WITH m AS (
            SELECT s.year, s.npi, l.excl_date, l.excl_type, l.rein_date, l.in_current_list, 'npi' AS match_type
            FROM mart.provider_scores s JOIN stg.leie l ON l.npi = s.npi
            UNION ALL
            SELECT s.year, s.npi, l.excl_date, l.excl_type, l.rein_date, l.in_current_list, 'name_state'
            FROM mart.provider_scores s
            JOIN stg.leie l ON l.npi IS NULL AND l.last_name = s.last_name
                           AND l.first_name = s.first_name AND l.state = s.state
        ), first_excl AS (
            SELECT year, npi, match_type, min(excl_date) AS excl_date, arg_min(excl_type, excl_date) AS excl_type,
                   arg_min(rein_date, excl_date) AS rein_date,
                   arg_min(in_current_list, excl_date) AS in_current
            FROM m GROUP BY ALL
        )
        SELECT s.*,
               n.excl_date AS excl_date_npi,  n.excl_type AS excl_type_npi,  n.rein_date AS rein_date_npi, n.in_current AS cur_npi,
               x.excl_date AS excl_date_name, x.excl_type AS excl_type_name, x.rein_date AS rein_date_name, x.in_current AS cur_name
        FROM mart.provider_scores s
        LEFT JOIN first_excl n ON n.year = s.year AND n.npi = s.npi AND n.match_type = 'npi'
        LEFT JOIN first_excl x ON x.year = s.year AND x.npi = s.npi AND x.match_type = 'name_state'
    """).df()
    con.close()

    ye = pd.to_datetime(df["year"].astype(str) + "-12-31")
    end = ye + pd.DateOffset(years=v["horizon_years"])
    npi_d, name_d = pd.to_datetime(df["excl_date_npi"]), pd.to_datetime(df["excl_date_name"])
    use_name = v["name_state_fallback"]
    # excluded and not yet reinstated by the end of Y -> not a prediction. A provider excluded years ago and
    # reinstated before Y (found through the cumulative history) is billing legitimately and stays in.
    # Still on the current list = never reinstated; off it = reinstated, at the recorded date if known
    # (monthly supplements) or at an unknown date (archived snapshots only), treated as before Y.
    def active(d, rein, cur):
        return (d <= ye) & (cur.fillna(False).astype(bool) | (pd.to_datetime(rein) > ye))
    npi_active = active(npi_d, df["rein_date_npi"], df["cur_npi"])
    name_active = active(name_d, df["rein_date_name"], df["cur_name"])
    already = npi_active | (name_active if use_name else False)
    df, ye, end, npi_d, name_d = df[~already], ye[~already], end[~already], npi_d[~already], name_d[~already]

    df = df.copy()
    df["label"] = ((npi_d > ye) & (npi_d <= end)).astype(int)
    df["label_fraud"] = (df["label"].eq(1) & df["excl_type_npi"].isin(v["fraud_related_types"])).astype(int)
    name_hit = ((name_d > ye) & (name_d <= end)) if use_name else False
    df["label_name"] = (df["label"].eq(1) | name_hit).astype(int)
    return df


def _fast_auc(inv: np.ndarray, n_groups: int, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted ROC AUC (ties count 1/2) from precomputed score ranks; O(n) per call."""
    wp = np.bincount(inv, weights=w * y, minlength=n_groups)
    wn = np.bincount(inv, weights=w * (1 - y), minlength=n_groups)
    P, N = wp.sum(), wn.sum()
    if P == 0 or N == 0:
        return np.nan
    neg_below = np.cumsum(wn) - wn
    return float((wp * (neg_below + 0.5 * wn)).sum() / (P * N))


def _with_ci(d: pd.DataFrame, label: str, reps: int, seed: int) -> dict:
    y = d[label].to_numpy().astype(float)
    ranks = {s: np.unique(d[s].to_numpy(), return_inverse=True) for s in SCORES}
    tiers = {t: d["risk_pct"].to_numpy() >= q for t, q in TIERS}

    def stats(w: np.ndarray) -> dict:
        pos = (y * w).sum()
        base = pos / w.sum()
        out = {f"auc_{s}": _fast_auc(inv, len(u), y, w) for s, (u, inv) in ranks.items()}
        for t, m in tiers.items():
            prec = (y[m] * w[m]).sum() / w[m].sum() if w[m].sum() else np.nan
            out[f"{t}_precision"], out[f"{t}_lift"] = prec, (prec / base if base else np.nan)
            out[f"{t}_recall"] = (y[m] * w[m]).sum() / pos if pos else np.nan
        return out

    point = stats(np.ones(len(d)))
    res = {"provider_years": int(len(d)), "providers": int(d["npi"].nunique()),
           "positive_provider_years": int(y.sum()),
           "positive_providers": int(d.loc[d[label] == 1, "npi"].nunique()),
           "base_rate": float(y.mean())}
    if res["positive_providers"] < 3 or reps == 0:
        return res | {k: {"est": float(v)} for k, v in point.items()}
    codes, uniq = pd.factorize(d["npi"])
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(reps) for k in point}
    for i in range(reps):
        # each provider drawn k times contributes weight k to all of its rows
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        for k, v in stats(counts[codes].astype(float)).items():
            draws[k][i] = v
    for k, v in point.items():
        lo, hi = np.nanpercentile(draws[k], [2.5, 97.5])
        res[k] = {"est": float(v), "ci95": [float(lo), float(hi)]}
    return res


def export_dashboard(df: pd.DataFrame, cfg: Config) -> None:
    """Compact file for the Streamlit app: needed columns only, float32, categoricals."""
    from .score import METRICS
    cols = ["year", "npi", "specialty", "state", "last_name", "first_name", "tot_benes", "risk_pct", "tier",
            "top_drivers", "n_flags", "label"] + list(METRICS) + [f"{m}__{sfx}" for m in METRICS for sfx in ("adj", "rz", "pct")]
    d = df[[c for c in cols if c in df.columns]].copy()
    for c in d.select_dtypes("float64").columns:
        d[c] = d[c].astype("float32")
    for c in ("specialty", "state", "tier"):
        d[c] = d[c].astype("category")
    d["year"] = d["year"].astype("int16")
    d.to_parquet(cfg.out_dir / "dashboard.parquet", index=False, row_group_size=50_000)


def run(cfg: Config) -> dict:
    v = cfg["validation"]
    reps, seed = int(v.get("bootstrap_reps", 500)), cfg["scoring"]["random_state"]
    df = labelled(cfg)
    res = {
        "definition": f"exclusion within {v['horizon_years']} years after the data year; "
                      "CIs = 95% provider-clustered bootstrap",
        "primary_npi_match": _with_ci(df, "label", reps, seed),
        "fraud_related_npi_match": _with_ci(df, "label_fraud", reps, seed),
        "sensitivity_npi_or_name_state": _with_ci(df, "label_name", reps, seed),
        "by_specialty": {s: _with_ci(g, "label", reps, seed) for s, g in df.groupby("specialty")},
        "by_year": {int(y): _with_ci(g, "label", 0, seed) for y, g in df.groupby("year")},
    }
    (cfg.reports_dir / "validation.json").write_text(json.dumps(res, indent=2))
    df.to_parquet(cfg.out_dir / "provider_scores_labelled.parquet", index=False)
    export_dashboard(df, cfg)
    p = res["primary_npi_match"]

    def fmt(k):
        e = p[k]
        return f"{e['est']:.2f} [{e['ci95'][0]:.2f}-{e['ci95'][1]:.2f}]" if "ci95" in e else f"{e['est']:.2f}"
    log.info("NPI label: %d providers excluded later (of %d) | AUC %s | top1%% lift %s | top5%% lift %s",
             p["positive_providers"], p["providers"], fmt("auc_risk_score"), fmt("top1_lift"), fmt("top5_lift"))
    return res
