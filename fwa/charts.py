"""Static figures for the write-up (PNG, light theme). No provider is named or identifiable."""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .score import METRICS

log = logging.getLogger(__name__)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SHORT = {"Interventional Pain Management": "Interventional Pain Mgmt",
         "Pain Management": "Pain Management",
         "Physical Therapist in Private Practice": "Physical Therapist"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def _save(fig, cfg, name):
    d = cfg.reports_dir / "figures"
    d.mkdir(exist_ok=True)
    fig.savefig(d / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("figure %s", name)


def peer_distribution(df: pd.DataFrame, cfg: Config, specialty: str, metric: str, year: int, legend_loc="upper left"):
    """The bell-curve view: log-scale peer distribution, normal fit, robust threshold, later-excluded providers."""
    col = f"{metric}__adj" if f"{metric}__adj" in df else metric      # the values the scorer used
    g = df[(df.specialty == specialty) & (df.year == year)].dropna(subset=[col])
    raw = g[col]
    x = np.log10(raw.clip(lower=1e-6))
    # threshold on the same (raw) scale the scorer uses, then drawn on the log axis
    raw_med = raw.median()
    raw_mad = 1.4826 * (raw - raw_med).abs().median()
    med, thr = np.log10(raw_med), np.log10(raw_med + cfg["scoring"]["z_flag"] * raw_mad)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(x, bins=80, color=BLUE, alpha=0.85, edgecolor=SURFACE, linewidth=0.5, density=True)
    xs = np.linspace(x.min(), x.max(), 400)
    ax.plot(xs, np.exp(-0.5 * ((xs - x.mean()) / x.std()) ** 2) / (x.std() * np.sqrt(2 * np.pi)),
            color=INK2, lw=1.5, ls="--", label="Normal fit to log values")
    ax.axvline(med, color=INK2, lw=1)
    ax.axvline(thr, color=ORANGE, lw=2)
    ymax = ax.get_ylim()[1]
    ax.text(med, ymax * 1.02, " median", color=INK2, va="top", fontsize=9)
    ax.text(thr, ymax * 1.02, f" robust z = {cfg['scoring']['z_flag']}", color=INK, va="top", fontsize=9)
    ex = x[g["label"] == 1]
    ax.scatter(ex, np.full(len(ex), ymax * 0.03), s=60, color=ORANGE, edgecolor=SURFACE, linewidth=2,
               zorder=5, label=f"Excluded by OIG within {cfg['validation']['horizon_years']} yrs (n={len(ex)})")
    ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200) if x.min() <= np.log10(t) <= x.max()]
    ax.set_xticks(np.log10(ticks), [str(t) for t in ticks])
    ax.set_xlabel(f"{METRICS[metric]}, volume-adjusted (log scale)" if col != metric else f"{METRICS[metric]} (log scale)")
    ax.set_ylabel("Density")
    ax.set_title(f"{SHORT.get(specialty, specialty)}, {year}: peer distribution (n={len(g):,} providers)")
    ax.set_ylim(0, ymax * 1.25)
    ax.legend(frameon=False, loc=legend_loc)
    _save(fig, cfg, f"peer_distribution_{metric}.png")


def gains_curve(df: pd.DataFrame, cfg: Config, reps: int = 300):
    """Share of later-excluded providers captured when reviewing the top X% by risk score."""
    d = df.sort_values("risk_pct", ascending=False).reset_index(drop=True)
    grid = np.linspace(0, 1, 101)
    y = d["label"].to_numpy(float)
    codes, uniq = pd.factorize(d["npi"])
    n = len(d)

    def curve(w):
        cw, cy = np.cumsum(w) / w.sum(), np.cumsum(w * y) / (w * y).sum()
        return np.interp(grid, np.r_[0, cw], np.r_[0, cy])

    rng = np.random.default_rng(cfg["scoring"]["random_state"])
    boots = np.array([curve(np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes].astype(float))
                      for _ in range(reps)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    est = curve(np.ones(n))
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.fill_between(grid * 100, lo * 100, hi * 100, color=BLUE, alpha=0.18, linewidth=0, label="95% CI")
    ax.plot(grid * 100, est * 100, color=BLUE, lw=2, label="Risk score")
    ax.plot([0, 100], [0, 100], color=INK2, lw=1, ls="--", label="Random review")
    for q in (5, 20):
        v = est[q] * 100
        ax.scatter([q], [v], s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.annotate(f"top {q}% → {v:.0f}%", (q, v), xytext=(8, -4), textcoords="offset points", fontsize=9)
    ax.set_xlim(0, 100), ax.set_ylim(0, 100)
    ax.set_xlabel("Provider-years reviewed, highest risk first (%)")
    ax.set_ylabel("Later OIG exclusions captured (%)")
    ax.set_title("Gains: exclusions captured vs review effort")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, cfg, "gains_curve.png")


def auc_forest(cfg: Config):
    r = json.loads((cfg.reports_dir / "validation.json").read_text())
    rows = [("All (NPI match)", r["primary_npi_match"]),
            ("Fraud-related exclusion types", r["fraud_related_npi_match"]),
            ("Adds name + state matches", r["sensitivity_npi_or_name_state"])]
    rows += [(SHORT.get(k, k), v) for k, v in r["by_specialty"].items()]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, (name, m) in enumerate(rows[::-1]):
        e = m["auc_risk_score"]
        ax.plot(e.get("ci95", [e["est"], e["est"]]), [i, i], color=BLUE, lw=2, solid_capstyle="round")
        ax.scatter([e["est"]], [i], s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.text(1.005, i, f"{e['est']:.2f}  ({m['positive_providers']} excluded)", va="center", fontsize=9, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [n for n, _ in rows[::-1]])
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUC with 95% CI (0.5 = no better than chance)")
    ax.set_title("Does the risk score rank later-excluded providers higher?")
    ax.grid(axis="y", visible=False)
    _save(fig, cfg, "auc_by_group.png")


def driver_frequency(df: pd.DataFrame, cfg: Config):
    """Which metric most often drives a top-5% score, per specialty (small multiples, shared scale)."""
    top = df[df["risk_pct"] >= 0.95]
    specs = list(SHORT)
    data = {}
    for s in specs:
        t = top[top.specialty == s]
        first = t["top_drivers"].dropna().str.split("; ").str[0].str.rsplit(" (", n=1).str[0]
        data[s] = (first.value_counts().head(5)[::-1] / max(len(t), 1) * 100, len(t))
    xmax = max(c.max() for c, _ in data.values()) * 1.2
    fig, axes = plt.subplots(len(specs), 1, figsize=(8, 8), sharex=True)
    for ax, s in zip(axes, specs):
        c, n = data[s]
        ax.barh(c.index, c.values, color=BLUE, height=0.6)
        for yv, v in enumerate(c.values):
            ax.text(v + xmax * 0.01, yv, f"{v:.0f}%", va="center", fontsize=9, color=INK2)
        ax.set_title(f"{SHORT[s]} (top-5% provider-years: {n:,})", fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, xmax)
    axes[-1].set_xlabel("Share of top-5% provider-years where this is the top driver (%)")
    fig.suptitle("What drives a top-5% risk score", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    _save(fig, cfg, "top_drivers.png")


def run(cfg: Config):
    _style()
    df = pd.read_parquet(cfg.out_dir / "provider_scores_labelled.parquet")
    latest = int(df["year"].max()) - 1          # a year with a longer observed follow-up window
    peer_distribution(df, cfg, "Physical Therapist in Private Practice", "srvcs_per_bene", latest)
    peer_distribution(df, cfg, "Interventional Pain Management", "code_intensity_index", latest, "upper right")
    gains_curve(df, cfg)
    auc_forest(cfg)
    driver_frequency(df, cfg)
