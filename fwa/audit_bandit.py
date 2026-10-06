"""Audit policy that learns from its own audit results (contextual bandit, one-step reinforcement learning).

`audit.py` plans each year's audits from a model fitted once. Here the plan is re-made every quarter and the model
is updated with what the audits found. Only audited providers reveal a result, which is what makes this a bandit
problem: the policy has to balance auditing providers it's confident about against learning about the rest.

Why simulated audit findings
----------------------------
The only observed outcome in public data is a later OIG exclusion. Over the held-out years 2020-2023 there are 62
later-excluded provider-years among 285,039: about 4 per quarter, and 0.84 on average inside a quarter's audit
list. A policy can't learn from a signal that sparse; any difference between policies would be noise.
Audits also find overpayments, unsupported units and upcoding that never reach exclusion. Those findings
aren't public, so they're simulated here from the billing outliers the score measures, through hidden
weights the policy doesn't see. Later-excluded providers are treated as near-certain findings.

Reward of auditing provider i (standardized Medicare dollars recovered):
    payment_i x (0.25 x later_excluded_i + 0.10 x finding_i)
    finding_i ~ Bernoulli(q_i), logit q_i = specialty intercept + hidden weights x clipped robust z-scores
From 2022 a new scheme appears: billing of passive / unattended modalities starts to predict findings (its hidden
weight rises), so a model frozen on earlier audits misses it.

Protocol
--------
- History: in 2016-2019 audits covered the top 5% by risk rank plus a 1% random sample; their (simulated) results
  form the log that both learning policies start from.
- Rounds: each held-out year's providers are split at random into 4 quarterly cohorts (16 rounds). Each round's
  hour budget is what auditing the top 5% of the cohort by rank would take; no back-to-back audits of a provider.
- Policies, selected greedily by value per audit hour within the budget:
    rank      audit down the risk ranking (the current review list)
    static    Bayesian logistic model of "finding" fitted on the history log, never updated
    thompson  the same model updated after every round (Laplace approximation), acting on a posterior draw
    thompson_forget  the same with old evidence fading (precision decays 20% a quarter); added after the
              primary run showed plain Thompson sampling adapting slowly, and reported as such
    oracle    knows each provider's true finding probability and exclusion (the ceiling)
- Scored with expected recovery (true q_i, observed exclusion labels), 5 seeds. Settings were fixed before the
  held-out run and not tuned on it.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import audit
from .config import Config

log = logging.getLogger(__name__)

TRUTH = {                      # hidden weights on clipped robust z-scores (the policy never sees these)
    "srvcs_per_bene": 0.35, "code_intensity_index": 0.30, "mue_max_ratio": 0.30,
    "timed_units_per_code_day": 0.25, "service_days_per_bene": 0.20, "pymt_per_bene_risk_adj": 0.15,
    "passive_modality_share": 0.0,
}
SHIFT = {"from_year": 2022, "feature": "passive_modality_share", "weight": 0.60}
INTERCEPT = -4.0               # about 2% of typical providers have a finding
R_EXCL, R_FIND = 0.25, 0.10
FORGET = 0.8                   # thompson_forget: per-quarter decay of posterior precision toward the prior
FEATS = list(TRUTH)


def _z(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([df[f"{f}__rz"].astype(float).fillna(0).clip(0, 6).to_numpy() for f in FEATS])


def true_q(df: pd.DataFrame, shift_weight: float = SHIFT["weight"]) -> np.ndarray:
    w = np.array([TRUTH[f] for f in FEATS])
    shift = df["year"].to_numpy() >= SHIFT["from_year"]
    z = _z(df)
    logit = INTERCEPT + z @ w + shift * shift_weight * z[:, FEATS.index(SHIFT["feature"])]
    q = 1 / (1 + np.exp(-logit))
    return np.where(df["label"].to_numpy() == 1, np.maximum(q, 0.9), q)


def design(df: pd.DataFrame, specialties: list[str]) -> np.ndarray:
    """What the policy sees: specialty, clipped robust z-scores, log payment, calibrated exclusion risk."""
    p = df["p_excl"].clip(1e-5, 1 - 1e-5).to_numpy()
    return np.column_stack([(df["specialty"] == s).to_numpy(float) for s in specialties]
                           + [_z(df) / 3, np.log1p(df["tot_stdzd_pymt"].to_numpy()) / 10 - 1,
                              np.log(p / (1 - p)) / 10])


class BayesLogit:
    """Bayesian logistic regression with a Gaussian (Laplace) posterior, updated batch by batch."""

    def __init__(self, dim: int, prior_var: float = 4.0):
        self.m = np.zeros(dim)
        self.H = np.eye(dim) / prior_var

    def update(self, X: np.ndarray, y: np.ndarray, iters: int = 25) -> None:
        m0, H0, w = self.m.copy(), self.H.copy(), self.m.copy()
        for _ in range(iters):                                  # Newton steps on the log posterior
            p = 1 / (1 + np.exp(-X @ w))
            g = X.T @ (y - p) - H0 @ (w - m0)
            H = H0 + (X * (p * (1 - p))[:, None]).T @ X
            step = np.linalg.solve(H, g)
            w = w + step
            if np.abs(step).max() < 1e-6:
                break
        p = 1 / (1 + np.exp(-X @ w))
        self.m, self.H = w, H0 + (X * (p * (1 - p))[:, None]).T @ X

    def prob(self, X, rng=None) -> np.ndarray:
        """Posterior mean, or (with rng) one posterior draw per provider, not one shared draw per round."""
        if rng is None:
            return 1 / (1 + np.exp(-X @ self.m))
        cov = np.linalg.inv(self.H)
        L = np.linalg.cholesky((cov + cov.T) / 2)
        w = self.m[None, :] + rng.standard_normal((len(X), len(self.m))) @ L.T
        return 1 / (1 + np.exp(-np.einsum("ij,ij->i", X, w)))


def greedy(value: np.ndarray, hours: np.ndarray, budget: float, blocked: np.ndarray) -> np.ndarray:
    return audit._fill(np.argsort(-(value / hours), kind="stable"), hours, budget, blocked)


def run_seed(df: pd.DataFrame, specialties, test_years, seed: int, shift_weight: float = SHIFT["weight"]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = design(df, specialties)
    q = true_q(df, shift_weight)
    pay, hours, y_ex = df["tot_stdzd_pymt"].to_numpy(), df["hours"].to_numpy(), df["label"].to_numpy()
    finding = (rng.random(len(df)) < q).astype(float)           # what an audit would reveal
    hit = np.maximum(finding, y_ex)
    expected = pay * (R_EXCL * y_ex + R_FIND * q)               # used for scoring only

    # history log 2016-2019: top 5% by rank plus a 1% random sample
    hist = df["year"].lt(min(test_years)).to_numpy()
    logged = hist & ((df["risk_pct"].to_numpy() >= 0.95) | (rng.random(len(df)) < 0.01))
    static = BayesLogit(X.shape[1])
    static.update(X[logged], hit[logged])
    ts = BayesLogit(X.shape[1])
    ts.m, ts.H = static.m.copy(), static.H.copy()
    tf = BayesLogit(X.shape[1])                                  # same, but old evidence fades each quarter
    tf.m, tf.H = static.m.copy(), static.H.copy()
    H_prior = np.eye(X.shape[1]) / 4.0
    avg_rec = R_FIND                                             # value = P(finding) x payment x recovery rate

    rows, prev, rnd = [], {p: set() for p in ("rank", "static", "thompson", "thompson_forget", "oracle")}, 0
    for yr in test_years:
        idx_year = np.where(df["year"].to_numpy() == yr)[0]
        quarter = rng.integers(0, 4, len(idx_year))
        for qt in range(4):
            idx = idx_year[quarter == qt]
            g_rank = df["risk_pct"].to_numpy()[idx]
            budget = hours[idx][g_rank >= np.quantile(g_rank, 0.95)].sum()
            npi = df["npi"].to_numpy()[idx]
            vals = {"rank": g_rank * 1.0,
                    "static": static.prob(X[idx]) * pay[idx] * avg_rec,
                    "thompson": ts.prob(X[idx], rng) * pay[idx] * avg_rec,
                    "thompson_forget": tf.prob(X[idx], rng) * pay[idx] * avg_rec,
                    "oracle": expected[idx]}
            for p, v in vals.items():
                blocked = np.isin(npi, list(prev[p]))
                if p == "rank":
                    sel = audit._fill(np.argsort(-v, kind="stable"), hours[idx], budget, blocked)
                else:
                    sel = greedy(v, hours[idx], budget, blocked)
                chosen = idx[sel]
                prev[p] = set(npi[sel])
                rows.append({"seed": seed, "round": rnd, "year": yr, "quarter": qt + 1, "policy": p,
                             "audits": int(sel.sum()), "recovered": float(expected[chosen].sum()),
                             "excluded_caught": int(y_ex[chosen].sum()),
                             "findings_observed": int(hit[chosen].sum())})
                if p == "thompson":
                    ts.update(X[chosen], hit[chosen])
                if p == "thompson_forget":
                    tf.H = FORGET * tf.H + (1 - FORGET) * H_prior
                    tf.update(X[chosen], hit[chosen])
            rnd += 1
    return pd.DataFrame(rows)


def run(cfg: Config) -> dict:
    s = audit.settings(cfg)
    df = audit.load(cfg)
    for c in [f"{f}__rz" for f in FEATS]:
        df[c] = pd.read_parquet(cfg.out_dir / "provider_scores_labelled.parquet", columns=[c])[c].to_numpy()
    specialties = sorted(df["specialty"].unique())
    years = sorted(df["year"].unique())
    test_years = [y for y in s["test_years"] if y in years]
    if not test_years or min(test_years) <= min(years):     # e.g. synthetic data: first half is the history
        test_years = years[len(years) // 2:]
    train = df[df["year"] < min(test_years)]
    model = audit.calibrate(train, specialties)
    df["p_excl"] = model.predict_proba(audit._features(df, specialties))[:, 1]
    df["hours"] = audit.audit_hours(df, s["cost"])
    df = df[df["year"] <= max(test_years)].reset_index(drop=True)

    seeds = cfg.raw.get("audit_bandit", {}).get("seeds", [1, 2, 3, 4, 5])
    t = pd.concat([run_seed(df, specialties, test_years, sd) for sd in seeds], ignore_index=True)
    t.to_csv(cfg.reports_dir / "audit_bandit_rounds.csv", index=False)

    tot = t.groupby(["seed", "policy"])[["recovered", "excluded_caught", "findings_observed", "audits"]].sum()
    per = tot["recovered"].unstack()
    test = df[df["year"].isin(test_years)]
    q_test = true_q(test)
    res = {
        "why_simulated": {
            "test_provider_years": len(test), "later_excluded": int(test["label"].sum()),
            "later_excluded_per_quarter": float(test["label"].sum() / (4 * len(test_years))),
            "expected_findings_per_quarter_all_providers": float(q_test.sum() / (4 * len(test_years))),
            "observed_hits_per_quarter_rank_list": float(t[t.policy == "rank"]["findings_observed"].mean()),
            "excluded_caught_per_quarter_rank_list": float(t[t.policy == "rank"]["excluded_caught"].mean()),
        },
        "seeds": seeds, "rounds": 4 * len(test_years),
        "totals_mean": {p: {k: float(tot.xs(p, level="policy")[k].mean()) for k in tot.columns}
                        for p in per.columns},
        "recovered_by_seed": per.round(0).to_dict(orient="index"),
        "thompson_vs_static_pct_by_seed": (per["thompson"] / per["static"] - 1).round(4).to_dict(),
        "forget_vs_static_pct_by_seed": (per["thompson_forget"] / per["static"] - 1).round(4).to_dict(),
        "thompson_vs_rank_pct_by_seed": (per["thompson"] / per["rank"] - 1).round(4).to_dict(),
        "static_vs_rank_pct_by_seed": (per["static"] / per["rank"] - 1).round(4).to_dict(),
    }
    # sensitivity (added after the primary run): how much learning is worth as the new scheme grows stronger
    after = t["year"] >= SHIFT["from_year"]
    st = t[after & (t.policy == "static")]["recovered"].sum()
    for p in ("thompson", "thompson_forget"):
        res[f"primary_after_shift_{p}_vs_static_pct"] = float(t[after & (t.policy == p)]["recovered"].sum() / st - 1)
    res["sensitivity_shift_weight"] = {}
    for sw in (1.2, 2.0):
        u = pd.concat([run_seed(df, specialties, test_years, sd, sw) for sd in seeds], ignore_index=True)
        a = u[u["year"] >= SHIFT["from_year"]]
        by = a.groupby(["seed", "policy"])["recovered"].sum().unstack()
        res["sensitivity_shift_weight"][str(sw)] = {
            "after_shift_thompson_vs_static_pct_by_seed": (by["thompson"] / by["static"] - 1).round(4).to_dict(),
            "after_shift_forget_vs_static_pct_by_seed": (by["thompson_forget"] / by["static"] - 1).round(4).to_dict(),
            "after_shift_share_of_oracle": {p: float((by[p] / by["oracle"]).mean()) for p in by.columns}}
    (cfg.reports_dir / "audit_bandit.json").write_text(json.dumps(res, indent=2, default=float))
    _chart(t, cfg)
    log.info("audit bandit: %s", {p: round(v["recovered"] / 1e6, 2) for p, v in res["totals_mean"].items()})
    return res


def _chart(t: pd.DataFrame, cfg: Config) -> None:
    import matplotlib.pyplot as plt

    from .charts import BLUE, INK2, ORANGE, _save, _style
    _style()
    m = t.groupby(["policy", "round"])["recovered"].mean().unstack(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    lab = {"rank": ("Audit down the risk ranking", INK2, "-"), "static": ("Model fitted once", ORANGE, "--"),
           "thompson": ("Thompson sampling (learns each quarter)", BLUE, "-"),
           "thompson_forget": ("Thompson sampling, old evidence fades", "#7a9e3b", "-"),
           "oracle": ("Oracle", "#0b0b0b", ":")}
    for p, (l, c, ls) in lab.items():
        axes[0].plot(m.index + 1, m[p].cumsum() / 1e6, color=c, ls=ls, label=l)
    share = m.div(m["oracle"], axis=0)
    for p in ("rank", "static", "thompson", "thompson_forget"):
        l, c, ls = lab[p]
        axes[1].plot(share.index + 1, 100 * share[p], color=c, ls=ls, marker="o", ms=3, label=l)
    shift_round = 4 * (SHIFT["from_year"] - int(t["year"].min())) + 0.5
    for ax in axes:
        ax.axvline(shift_round, color="#b9b7b1", lw=1)
        ax.set_xlabel("Quarter (2020 Q1 = 1)")
        ax.set_xticks([1, 4, 8, 12, 16])
    axes[0].set_title("Cumulative expected recovery")
    axes[0].set_ylabel("$ millions (standardized)")
    axes[1].set_title("Recovery per quarter, % of oracle")
    axes[1].set_ylabel("%")
    axes[1].text(shift_round + 0.3, axes[1].get_ylim()[0] + 2, "new billing scheme", fontsize=8, color=INK2)
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, cfg, "audit_bandit.png")
