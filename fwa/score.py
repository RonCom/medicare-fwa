"""Peer-group outlier scoring.

Peer group = (year, specialty). For each metric we compute:
  * classic z-score and percentile  -> the bell-curve view
  * robust z = (x - median) / (1.4826 * MAD) -> resistant to the heavy right tails in claims data
A provider's rule-based composite is the mean of its k largest positive robust z's.
An Isolation Forest per peer group adds a multivariate score (unusual *combinations* of metrics).
Final risk score = weighted average of the two within-group percentiles (rules weighted higher:
they are explainable and, on 2021-24 data, separated later exclusions better).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import Config
from .load import connect

log = logging.getLogger(__name__)

# metric -> human label. All are "higher = more unusual".
METRICS = {
    "srvcs_per_bene": "Services per beneficiary",
    "pymt_per_bene_risk_adj": "Risk-adjusted payment per beneficiary",
    "em_high_share": "Share of level 4-5 office visits",
    "em_lvl5_share": "Share of level 5 office visits",
    "code_intensity_index": "Code intensity vs peers (payment-weighted)",
    "max_code_intensity": "Most extreme single-code intensity",
    "timed_units_per_code_day": "Timed (15-min) units per patient-day",
    "service_days_per_bene": "Service days per beneficiary",
    "code_concentration_hhi": "Concentration of payment in few codes",
    "passive_modality_share": "Share of passive / unattended modalities",
    "mue_max_ratio": "Units per patient-day vs NCCI MUE limit",
    "opioid_claim_share": "Opioid share of Part D claims",
    "opioid_la_share": "Long-acting share of opioid claims",
    "brand_claim_share": "Brand-name share of Part D claims",
    "drug_cost_per_bene": "Part D cost per beneficiary",
}
MAD_K = 1.4826

# Volume behind each metric: small denominators make rates noisy, so they are shrunk harder.
DENOM = {
    "srvcs_per_bene": "tot_benes", "pymt_per_bene_risk_adj": "tot_benes", "service_days_per_bene": "tot_benes",
    "code_intensity_index": "tot_benes", "max_code_intensity": "tot_benes",
    "em_high_share": "em_visits", "em_lvl5_share": "em_visits",
    "timed_units_per_code_day": "timed_code_days",
    "code_concentration_hhi": "tot_srvcs", "passive_modality_share": "tot_srvcs",
    "opioid_claim_share": "rx_claims", "brand_claim_share": "rx_claims", "drug_cost_per_bene": "rx_claims",
    "opioid_la_share": "opioid_claims",
    "mue_max_ratio": "tot_benes",
}


def shrink(x: pd.Series, n: pd.Series) -> tuple[pd.Series, float]:
    """Empirical-Bayes (Buhlmann credibility) shrinkage toward the peer median.

    Model: observed x = true rate + noise, with Var(x | n) = tau2 + sigma2 / n.
    tau2 (between-provider) and sigma2 (within-provider) are estimated by regressing squared
    deviations on 1/n (method of moments; x winsorized at the 99.5th pct for the fit only).
    Shrunk x = (n * x + k * median) / (n + k), with k = sigma2 / tau2: a provider with n = k
    volume is pulled halfway to the peer median. If the fit isn't valid (tau2 or sigma2 <= 0),
    no shrinkage is applied (k = 0).
    """
    ok = x.notna() & n.gt(0)
    if ok.sum() < 30:
        return x, 0.0
    xv, nv = x[ok], n[ok].astype(float)
    xc = xv.clip(upper=xv.quantile(0.995))
    mu = np.average(xc, weights=nv)
    sigma2, tau2 = np.polyfit(1.0 / nv, (xc - mu) ** 2, 1)
    if not (tau2 > 0 and sigma2 > 0):
        return x, 0.0
    k = sigma2 / tau2
    target = xv.median()
    out = x.copy()
    out[ok] = (nv * xv + k * target) / (nv + k)
    return out, float(k)


def _robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * MAD_K
    if not np.isfinite(mad) or mad == 0:
        mad = s.std(ddof=0)
    return (s - med) / mad if mad and np.isfinite(mad) else s * np.nan


def _score_group(g: pd.DataFrame, sc: dict) -> pd.DataFrame:
    g = g.copy()
    metrics = [m for m in METRICS if g[m].notna().sum() >= sc["min_peer_group"]]
    shrink_k = {}
    for m in metrics:
        x = g[m]
        if sc.get("shrinkage", True) and DENOM.get(m) in g:
            x, shrink_k[m] = shrink(x, g[DENOM[m]])
        g[f"{m}__adj"] = x
        g[f"{m}__z"] = (x - x.mean()) / x.std(ddof=0)
        g[f"{m}__rz"] = _robust_z(x)
        g[f"{m}__pct"] = x.rank(pct=True)
    rz = g[[f"{m}__rz" for m in metrics]].clip(lower=0, upper=10)
    k = sc["top_k_for_composite"]
    g["composite_z"] = rz.apply(lambda r: np.sort(r.dropna().values)[::-1][:k].mean() if r.notna().any() else 0.0, axis=1)
    g["n_flags"] = (rz > sc["z_flag"]).sum(axis=1)
    g["top_drivers"] = rz.apply(
        lambda r: "; ".join(f"{METRICS[c[:-4]]} ({v:.1f})" for c, v in r.dropna().sort_values(ascending=False).head(3).items() if v > 2),
        axis=1)

    # Isolation Forest only on metrics this peer group actually has (e.g. PTs have no E/M or Part D);
    # median-imputing a mostly-missing column adds noise, not signal.
    if_metrics = [m for m in metrics if g[m].notna().mean() >= sc["iforest_min_coverage"]]
    X = g[[f"{m}__adj" for m in if_metrics]].apply(lambda c: np.log1p(c.clip(lower=0))).apply(lambda c: c.fillna(c.median()))
    X = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=sc["isolation_forest_trees"], random_state=sc["random_state"])
    iso.fit(X)
    g["iforest_score"] = -iso.score_samples(X)
    g["iforest_metrics"] = len(if_metrics)

    g["composite_pct"] = g["composite_z"].rank(pct=True)
    g["iforest_pct"] = g["iforest_score"].rank(pct=True)
    w = sc["weights"]
    g["risk_score"] = (w["composite"] * g["composite_pct"] + w["iforest"] * g["iforest_pct"]) / (w["composite"] + w["iforest"])
    g["risk_pct"] = g["risk_score"].rank(pct=True)
    g["tier"] = np.select([g["risk_pct"] >= 0.99, g["risk_pct"] >= 0.95], ["Top 1%", "Top 5%"], "Other")
    g["peer_n"] = len(g)
    g.attrs["shrink_k"] = shrink_k
    return g


def score_frame(df: pd.DataFrame, sc: dict) -> pd.DataFrame:
    """Score a provider_features frame (from DuckDB or Snowflake); shrinkage k values in .attrs."""
    df = df.sort_values(["year", "specialty", "npi"]).reset_index(drop=True)   # same order everywhere
    parts, ks = [], []
    for (year, spec), g in df.groupby(["year", "specialty"]):
        if len(g) < sc["min_peer_group"]:
            log.warning("skip %s %s: only %d providers", year, spec, len(g))
            continue
        scored = _score_group(g, sc)
        ks += [{"year": int(year), "specialty": spec, "metric": m, "k": k,
                "median_volume": float(g[DENOM[m]].median())} for m, k in scored.attrs["shrink_k"].items()]
        parts.append(scored)
    scores = pd.concat(parts, ignore_index=True)
    scores.attrs["shrink_k"] = ks
    return scores


def run(cfg: Config) -> pd.DataFrame:
    con = connect(cfg)
    df = con.execute("SELECT * FROM mart.provider_features").df()
    scores = score_frame(df, cfg["scoring"])
    if scores.attrs.get("shrink_k"):
        pd.DataFrame(scores.attrs["shrink_k"]).to_csv(cfg.reports_dir / "shrinkage_k.csv", index=False)
    con.register("scores_df", scores)
    con.execute("CREATE OR REPLACE TABLE mart.provider_scores AS SELECT * FROM scores_df")
    con.close()
    scores.to_parquet(cfg.out_dir / "provider_scores.parquet", index=False)
    log.info("scored %d provider-years; %d in Top 1%%", len(scores), (scores["tier"] == "Top 1%").sum())
    return scores
