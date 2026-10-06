"""One evaluator for every score, so the baseline and each experiment are compared the same way.

Inputs: a score per (year, npi) and the labels from fwa.validate.labelled (same exclusion rule
for everyone). Metrics: AUC, top-5% lift and recall, and top-5% dollar recall (the share of
later-excluded providers' Medicare payments that falls in the top 5%, a cost-based view as argued
by Hamid et al. 2024). 95% CIs from a bootstrap that resamples providers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .validate import _fast_auc


def evaluate(scores: pd.DataFrame, labels: pd.DataFrame, score_col: str = "score", label_col: str = "label",
             within_group: bool = True, reps: int = 500, seed: int = 42, top: float = 0.95) -> dict:
    d = labels[["year", "npi", "specialty", label_col, "tot_stdzd_pymt"]].merge(
        scores[["year", "npi", score_col]], on=["year", "npi"], how="inner").dropna(subset=[score_col])
    # within-group: rank inside specialty x year, as the baseline does (raw scores aren't
    # comparable across specialties with different exclusion rates)
    d["s"] = d.groupby(["year", "specialty"])[score_col].rank(pct=True) if within_group else d[score_col]
    d["pct"] = d.groupby(["year", "specialty"])[score_col].rank(pct=True)
    y = d[label_col].to_numpy(float)
    pay = d["tot_stdzd_pymt"].fillna(0).to_numpy(float)
    u, inv = np.unique(d["s"].to_numpy(), return_inverse=True)
    m = d["pct"].to_numpy() >= top

    def stats(w):
        pos = (y * w).sum()
        base = pos / w.sum()
        prec = (y[m] * w[m]).sum() / w[m].sum()
        dol = (y * w * pay).sum()
        return {"auc": _fast_auc(inv, len(u), y, w), "top5_lift": prec / base if base else np.nan,
                "top5_recall": (y[m] * w[m]).sum() / pos if pos else np.nan,
                "top5_dollar_recall": (y[m] * w[m] * pay[m]).sum() / dol if dol else np.nan}

    point = stats(np.ones(len(d)))
    out = {"rows": int(len(d)), "providers": int(d["npi"].nunique()),
           "positive_providers": int(d.loc[d[label_col] == 1, "npi"].nunique())}
    if reps and out["positive_providers"] >= 3:
        codes, uniq = pd.factorize(d["npi"])
        rng = np.random.default_rng(seed)
        draws = {k: np.empty(reps) for k in point}
        for i in range(reps):
            w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes].astype(float)
            for k, v in stats(w).items():
                draws[k][i] = v
        for k, v in point.items():
            lo, hi = np.nanpercentile(draws[k], [2.5, 97.5])
            out[k] = {"est": float(v), "ci95": [float(lo), float(hi)]}
    else:
        out |= {k: {"est": float(v)} for k, v in point.items()}
    out["by_specialty_auc"] = {}
    for spec, g in d.groupby("specialty"):
        if 0 < g[label_col].sum() < len(g):
            uu, ii = np.unique(g["s"].to_numpy(), return_inverse=True)
            out["by_specialty_auc"][spec] = _fast_auc(ii, len(uu), g[label_col].to_numpy(float), np.ones(len(g)))
    return out


def fmt(e: dict, d: int = 2, pct: bool = False) -> str:
    if pct:
        s = f"{e['est'] * 100:.0f}%"
        return s + (f" [{e['ci95'][0] * 100:.0f}–{e['ci95'][1] * 100:.0f}%]" if "ci95" in e else "")
    s = f"{e['est']:.{d}f}"
    return s + (f" [{e['ci95'][0]:.{d}f}–{e['ci95'][1]:.{d}f}]" if "ci95" in e else "")


def compare(a: pd.DataFrame, b: pd.DataFrame, labels: pd.DataFrame, label_col: str = "label",
            reps: int = 1000, seed: int = 42, top: float = 0.95) -> dict:
    """Paired bootstrap of (b - a) on the same provider resamples: AUC and top-5% dollar recall.
    Both scores are ranked within specialty x year; rows scored by both are used."""
    d = labels[["year", "npi", "specialty", label_col, "tot_stdzd_pymt"]].merge(
        a[["year", "npi", "score"]].rename(columns={"score": "sa"}), on=["year", "npi"]).merge(
        b[["year", "npi", "score"]].rename(columns={"score": "sb"}), on=["year", "npi"]).dropna(subset=["sa", "sb"])
    y = d[label_col].to_numpy(float)
    pay = d["tot_stdzd_pymt"].fillna(0).to_numpy(float)
    prep = {}
    for k in ("sa", "sb"):
        pct = d.groupby(["year", "specialty"])[k].rank(pct=True).to_numpy()
        u, inv = np.unique(pct, return_inverse=True)
        prep[k] = (inv, len(u), pct >= top)

    def stats(w):
        out = {}
        for k, (inv, n, m) in prep.items():
            dol = (y * w * pay).sum()
            out[k] = (_fast_auc(inv, n, y, w), (y[m] * w[m] * pay[m]).sum() / dol if dol else np.nan)
        return out["sb"][0] - out["sa"][0], out["sb"][1] - out["sa"][1]

    est = stats(np.ones(len(d)))
    codes, uniq = pd.factorize(d["npi"])
    rng = np.random.default_rng(seed)
    draws = np.array([stats(np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes].astype(float))
                      for _ in range(reps)])
    res = {"rows": int(len(d)), "positive_providers": int(d.loc[d[label_col] == 1, "npi"].nunique())}
    for i, k in enumerate(("auc_diff", "dollar_recall_diff")):
        lo, hi = np.nanpercentile(draws[:, i], [2.5, 97.5])
        res[k] = {"est": float(est[i]), "ci95": [float(lo), float(hi)]}
    return res
