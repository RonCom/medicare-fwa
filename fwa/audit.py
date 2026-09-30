"""Audit planning: which flagged providers to review under a fixed audit-hour budget.

The risk score ranks providers; an investigations unit still has to decide whom to audit with limited staff hours.
This step turns the ranking into an audit plan and compares plans on held-out data years.

1. Calibrate: P(later exclusion) from the within-specialty risk percentile, fitted on the train years only
   (logistic regression, one intercept per specialty and a shared slope on logit(risk_pct)).
2. Value and cost: expected value = P(exclusion) x standardized Medicare payment (the dollars at stake);
   audit hours grow with patient panel size (config `audit.cost`).
3. Plan each year with an integer program (scipy / HiGHS):
       maximize   sum_i v_i x_i
       subject to sum_i c_i x_i <= budget hours
                  hours per specialty >= floor x its share of providers x budget   (coverage)
                  x_i = 0 if provider i was audited the previous year                (no back-to-back audits)
                  x_i in {0, 1}
4. Compare on held-out years against auditing straight down the risk ranking (current practice) and a
   greedy value-per-hour heuristic, at the same budget. Confidence intervals resample providers.

Budget: by default each year gets the hours needed to audit the top 5% of providers by risk percentile, so the
rank plan equals the project's existing "top 5%" review list and the comparison isolates the planning step.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.linear_model import LogisticRegression

from .config import Config

log = logging.getLogger(__name__)

COLS = ["year", "npi", "specialty", "tot_benes", "tot_stdzd_pymt", "risk_pct", "label"]
DEFAULTS = {
    "train_years": [2016, 2017, 2018, 2019],
    "test_years": [2020, 2021, 2022, 2023],     # 2024's 3-year label window is still open
    "budget_top_pct": 0.05,
    "sweep_top_pct": [0.01, 0.02, 0.05, 0.10],
    "specialty_floor": 0.25,
    "cost": {"base_hours": 12, "hours_per_doubling": 6, "bene_scale": 25, "max_hours": 60},
    "bootstrap_reps": 500,
}


def settings(cfg: Config) -> dict:
    s = {**DEFAULTS, **cfg.raw.get("audit", {})}
    s["cost"] = {**DEFAULTS["cost"], **s.get("cost", {})}
    return s


def load(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.out_dir / "provider_scores_labelled.parquet", columns=COLS)
    df["tot_stdzd_pymt"] = df["tot_stdzd_pymt"].astype(float).fillna(0.0)
    return df.reset_index(drop=True)


def audit_hours(df: pd.DataFrame, c: dict) -> np.ndarray:
    """Hours for one desk review: a fixed part plus more sampled claims for larger patient panels."""
    h = c["base_hours"] + c["hours_per_doubling"] * np.log2(1 + df["tot_benes"].to_numpy(float) / c["bene_scale"])
    return np.clip(h, c["base_hours"], c["max_hours"])


def _features(df: pd.DataFrame, specialties: list[str]) -> np.ndarray:
    p = df["risk_pct"].clip(1e-4, 1 - 1e-4).to_numpy(float)
    X = [np.log(p / (1 - p))]
    X += [(df["specialty"] == s).to_numpy(float) for s in specialties]
    return np.column_stack(X)


def calibrate(train: pd.DataFrame, specialties: list[str]) -> LogisticRegression:
    m = LogisticRegression(C=1e4, fit_intercept=False, max_iter=2000)   # specialty dummies act as intercepts
    return m.fit(_features(train, specialties), train["label"].to_numpy())


def _floor_hours(g: pd.DataFrame, budget: float, floor: float) -> dict:
    share = g["specialty"].value_counts(normalize=True)
    return {s: floor * share[s] * budget for s in share.index}


def _candidates(g: pd.DataFrame, budget: float, blocked: np.ndarray, mult: float = 3.0) -> np.ndarray:
    """Shrink the ILP to providers that could plausibly be chosen: within each specialty, the best value-per-hour
    providers covering `mult` x the budget. Anything below that cut-off is dominated many times over, and the
    reduction takes the solve from minutes to about a second per year."""
    keep = np.zeros(len(g), bool)
    ratio = (g["value"] / g["hours"]).to_numpy()
    c, spec = g["hours"].to_numpy(), g["specialty"].to_numpy()
    for s in np.unique(spec):
        idx = np.where((spec == s) & ~blocked)[0]
        idx = idx[np.argsort(-ratio[idx], kind="stable")]
        n = np.searchsorted(np.cumsum(c[idx]), mult * budget) + 1
        keep[idx[:n]] = True
    return keep


def plan_ilp(g: pd.DataFrame, budget: float, floor: float, blocked: np.ndarray) -> np.ndarray:
    cand = _candidates(g, budget, blocked)
    h = g[cand]
    v, c, spec = h["value"].to_numpy(), h["hours"].to_numpy(), h["specialty"].to_numpy()
    rows, lo, hi = [c], [0.0], [budget]
    for s, need in _floor_hours(g, budget, floor).items():
        if need > 0:
            rows.append(c * (spec == s))
            lo.append(need)
            hi.append(np.inf)
    n = len(h)
    res = milp(-v, constraints=LinearConstraint(np.vstack(rows), lo, hi), integrality=np.ones(n),
               bounds=Bounds(np.zeros(n), np.ones(n)), options={"mip_rel_gap": 1e-4, "time_limit": 60})
    if res.x is None:
        raise RuntimeError(f"audit ILP failed: {res.message}")
    x = np.zeros(len(g), bool)
    x[np.where(cand)[0][res.x > 0.5]] = True
    return x


def _fill(order: np.ndarray, c: np.ndarray, budget: float, blocked: np.ndarray) -> np.ndarray:
    """Walk down a priority order, auditing each provider that still fits in the remaining hours."""
    x = np.zeros(len(c), bool)
    left = budget
    for i in order:
        if not blocked[i] and c[i] <= left:
            x[i], left = True, left - c[i]
    return x


def plan_rank(g, budget, blocked):
    return _fill(np.argsort(-g["risk_pct"].to_numpy(), kind="stable"), g["hours"].to_numpy(), budget, blocked)


def plan_greedy(g, budget, blocked):
    return _fill(np.argsort(-(g["value"] / g["hours"]).to_numpy(), kind="stable"), g["hours"].to_numpy(), budget, blocked)


def simulate(df: pd.DataFrame, top_pct: float, floor: float, years: list[int]) -> pd.DataFrame:
    """Plan each year in order under every policy; returns one boolean column per policy."""
    out = df[df["year"].isin(years)].copy()
    prev = {p: set() for p in ("rank", "greedy", "ilp")}
    for p in prev:
        out[p] = False
    for y in sorted(years):
        idx = out.index[out["year"] == y]
        g = out.loc[idx]
        top = g["risk_pct"] >= g["risk_pct"].quantile(1 - top_pct)
        budget = float(g.loc[top, "hours"].sum())
        for p, done in prev.items():
            blocked = g["npi"].isin(done).to_numpy()
            if p == "rank":
                x = plan_rank(g, budget, blocked)
            elif p == "greedy":
                x = plan_greedy(g, budget, blocked)
            else:
                x = plan_ilp(g, budget, floor, blocked)
            out.loc[idx, p] = x
            prev[p] = set(g.loc[x, "npi"])
        out.loc[idx, "budget"] = budget
        log.info("year %s: budget %.0f h, audits rank %d / ilp %d", y, budget,
                 out.loc[idx, "rank"].sum(), out.loc[idx, "ilp"].sum())
    return out


def _metrics(d: pd.DataFrame, pol: str, w: np.ndarray) -> dict:
    x, y, pay = d[pol].to_numpy(bool), d["label"].to_numpy(float), d["tot_stdzd_pymt"].to_numpy()
    pos = (w * y).sum()
    return {"recall": (w * y * x).sum() / pos,
            "dollar_recall": (w * y * x * pay).sum() / (w * y * pay).sum(),
            "precision": (w * y * x).sum() / (w * x).sum(),
            "caught_per_1000h": 1000 * (w * y * x).sum() / (w * x * d["hours"].to_numpy()).sum()}


def evaluate(d: pd.DataFrame, reps: int, seed: int) -> dict:
    pols = ("rank", "greedy", "ilp")
    point = {p: _metrics(d, p, np.ones(len(d))) for p in pols}
    codes, uniq = pd.factorize(d["npi"])
    rng = np.random.default_rng(seed)
    draws = {p: {k: [] for k in point[p]} for p in pols}
    for _ in range(reps):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes].astype(float)
        for p in pols:
            for k, v in _metrics(d, p, w).items():
                draws[p][k].append(v)
    res = {"provider_years": len(d), "positive_provider_years": int(d["label"].sum()),
           "positive_providers": int(d.loc[d["label"] == 1, "npi"].nunique())}
    for p in pols:
        res[p] = {"audits": int(d[p].sum()), "hours": float((d[p] * d["hours"]).sum()),
                  "positives_caught": int(d.loc[d[p], "label"].sum()),
                  "by_specialty": {s: int(n) for s, n in d.loc[d[p], "specialty"].value_counts().items()}}
        for k, v in point[p].items():
            lo, hi = np.nanpercentile(draws[p][k], [2.5, 97.5])
            res[p][k] = {"est": float(v), "ci95": [float(lo), float(hi)]}
    for p in ("greedy", "ilp"):
        for k in ("recall", "dollar_recall"):
            diff = np.array(draws[p][k]) - np.array(draws["rank"][k])
            lo, hi = np.nanpercentile(diff, [2.5, 97.5])
            res[f"{p}_minus_rank_{k}"] = {"est": point[p][k] - point["rank"][k], "ci95": [float(lo), float(hi)]}
    return res


def calibration_table(d: pd.DataFrame, bins=(0, .5, .8, .9, .95, .99, 1.0001)) -> list[dict]:
    b = pd.cut(d["risk_pct"], bins, right=False)
    t = d.groupby(b, observed=True).agg(n=("label", "size"), observed=("label", "mean"), predicted=("p_excl", "mean"))
    return [{"risk_pct": str(i), "n": int(r.n), "observed": float(r.observed), "predicted": float(r.predicted)}
            for i, r in t.iterrows()]


def run(cfg: Config) -> dict:
    s = settings(cfg)
    seed = cfg["scoring"]["random_state"]
    df = load(cfg)
    specialties = sorted(df["specialty"].unique())
    years = sorted(df["year"].unique())
    if not set(s["train_years"]) & set(years) or not set(s["test_years"]) & set(years):
        half = len(years) // 2                      # e.g. synthetic data: first half trains, second half tests
        s["train_years"], s["test_years"] = years[:half], years[half:]
        log.warning("configured audit years not in data; train %s, test %s", s["train_years"], s["test_years"])
    s["train_years"] = [int(y) for y in s["train_years"] if y in years]
    s["test_years"] = [int(y) for y in s["test_years"] if y in years]
    train = df[df["year"].isin(s["train_years"])]
    model = calibrate(train, specialties)
    df["p_excl"] = model.predict_proba(_features(df, specialties))[:, 1]
    df["hours"] = audit_hours(df, s["cost"])
    df["value"] = df["p_excl"] * df["tot_stdzd_pymt"]
    test = df[df["year"].isin(s["test_years"])]

    main = simulate(df, s["budget_top_pct"], s["specialty_floor"], s["test_years"])
    res = {
        "setup": {k: s[k] for k in ("train_years", "test_years", "budget_top_pct", "specialty_floor", "cost")},
        "calibration": {"coef_logit_risk_pct": float(model.coef_[0][0]),
                        "brier_test": float(np.mean((test["p_excl"] - test["label"]) ** 2)),
                        "predicted_positives_test": float(test["p_excl"].sum()),
                        "observed_positives_test": int(test["label"].sum()),
                        "by_risk_pct_test": calibration_table(test)},
        "main": evaluate(main, s["bootstrap_reps"], seed),
        "sweep": {}, "floor_sensitivity": {},
    }
    for q in s["sweep_top_pct"]:
        d = main if q == s["budget_top_pct"] else simulate(df, q, s["specialty_floor"], s["test_years"])
        res["sweep"][str(q)] = {p: _metrics(d, p, np.ones(len(d))) for p in ("rank", "greedy", "ilp")}
    for f in (0.0, 0.5):
        d = simulate(df, s["budget_top_pct"], f, s["test_years"])
        res["floor_sensitivity"][str(f)] = _metrics(d, "ilp", np.ones(len(d))) | {
            "by_specialty": {k: int(v) for k, v in d.loc[d["ilp"].astype(bool), "specialty"].value_counts().items()}}
    (cfg.reports_dir / "audit_plan.json").write_text(json.dumps(res, indent=2))
    _chart(res, cfg)
    return res


def _chart(res: dict, cfg: Config) -> None:
    import matplotlib.pyplot as plt

    from .charts import BLUE, INK2, ORANGE, _save, _style
    _style()
    qs = sorted(res["sweep"], key=float)
    x = [100 * float(q) for q in qs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, k, title in ((axes[0], "recall", "Later-excluded providers audited"),
                         (axes[1], "dollar_recall", "Their Medicare dollars audited")):
        for p, col, lab in (("rank", INK2, "Audit down the risk ranking"), ("ilp", BLUE, "Optimized plan (ILP)"),
                            ("greedy", ORANGE, "Greedy value per hour")):
            ax.plot(x, [100 * res["sweep"][q][p][k] for q in qs], marker="o", color=col, label=lab,
                    ls="--" if p == "greedy" else "-")
        ax.set_title(title)
        ax.set_xlabel("Audit budget (hours to review the top N% by rank)")
        ax.set_ylabel("% of held-out total")
        ax.set_xticks(x, [f"{v:g}%" for v in x])
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, cfg, "audit_budget_frontier.png")
