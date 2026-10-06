"""Medicare provider outlier screening dashboard.

Run:  uv run streamlit run dashboard/app.py            (real data)
      FWA_DATA_DIR=data_synthetic uv run streamlit run dashboard/app.py

Providers are shown with a pseudonymous ID. Set FWA_SHOW_IDENTIFIERS=1 to show NPI and name
(local use only - don't publish identified results).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fwa.score import METRICS  # noqa: E402

DATA_DIR = ROOT / os.environ.get("FWA_DATA_DIR", "data")
REPORT_DIR = ROOT / "reports" / ("synthetic" if "synthetic" in DATA_DIR.name else "real")
SHOW_IDS = os.environ.get("FWA_SHOW_IDENTIFIERS") == "1"
Z_FLAG = 3.5

BLUE, ORANGE, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#52514e", "#e4e3df", "#fcfcfb"
SHORT = {"Interventional Pain Management": "Interventional Pain Mgmt",
         "Physical Therapist in Private Practice": "Physical Therapist"}

st.set_page_config(page_title="Medicare Outlier Screening", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading scores…")
def load() -> pd.DataFrame:
    path = DATA_DIR / "out" / "dashboard.parquet"
    if not path.exists():
        st.error(f"{path} not found. Run `uv run fwa validate` first.")
        st.stop()
    df = pd.read_parquet(path)
    df["provider_id"] = df["npi"].map(lambda n: "P-" + hashlib.sha1(f"fwa:{n}".encode()).hexdigest()[:8].upper())
    df["risk_pctile"] = (df["risk_pct"] * 100).round(1)
    if not SHOW_IDS:
        df = df.drop(columns=["last_name", "first_name"], errors="ignore")
    return df


@st.cache_data
def validation() -> dict | None:
    p = REPORT_DIR / "validation.json"
    return json.loads(p.read_text()) if p.exists() else None


def short(s: str) -> str:
    return SHORT.get(s, s)


def fig_layout(fig: go.Figure, height: int = 360, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor=SURFACE,
                      paper_bgcolor=SURFACE, font=dict(color="#0b0b0b", size=13), hoverlabel=dict(bgcolor="white"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


df = load()
val = validation()

# ---------------------------------------------------------------- sidebar filters
st.sidebar.title("Filters")
specs = sorted(df["specialty"].unique())
sel_specs = st.sidebar.multiselect("Specialty", specs, default=specs, format_func=short)
years = sorted(df["year"].unique())
sel_years = st.sidebar.multiselect("Data year", years, default=years)
states = sorted(df["state"].dropna().unique())
sel_states = st.sidebar.multiselect("State", states, placeholder="All states")
st.sidebar.caption("Scores compare each provider with same-specialty peers in the same year, nationally.")
if not SHOW_IDS:
    st.sidebar.info("Providers are pseudonymized. An outlier isn't evidence of fraud.")

f = df[df["specialty"].isin(sel_specs) & df["year"].isin(sel_years)]
if sel_states:
    f = f[f["state"].isin(sel_states)]

st.title("Medicare provider outlier screening")
st.caption("Public CMS 2016–2024 data · peer-benchmarked robust z-scores + Isolation Forest · "
           "validated against later HHS-OIG exclusions")

tab_over, tab_dist, tab_queue, tab_method = st.tabs(["Overview", "Peer distributions", "Review queue", "Method & caveats"])

# ---------------------------------------------------------------- overview
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Provider-years", f"{len(f):,}")
    c2.metric("Top 5% tier", f"{(f['risk_pct'] >= 0.95).sum():,}")
    c3.metric("Later excluded (NPI match)", f"{f.loc[f['label'] == 1, 'npi'].nunique():,} providers")
    if val:
        a = val["primary_npi_match"]["auc_risk_score"]
        ci = f" (95% CI {a['ci95'][0]:.2f}–{a['ci95'][1]:.2f})" if "ci95" in a else ""
        c4.metric("AUC, all specialties", f"{a['est']:.2f}", help="Validation over all data" + ci)

    left, right = st.columns(2)
    with left:
        st.subheader("Review effort vs exclusions captured")
        d = f.sort_values("risk_pct", ascending=False)
        if d["label"].sum():
            x = np.arange(1, len(d) + 1) / len(d) * 100
            y = d["label"].cumsum().to_numpy() / d["label"].sum() * 100
            idx = np.unique(np.linspace(0, len(d) - 1, 400).astype(int))
            fig = go.Figure()
            fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random review",
                            line=dict(color=INK2, dash="dash", width=1), hoverinfo="skip")
            fig.add_scatter(x=x[idx], y=y[idx], mode="lines", name="Risk score", line=dict(color=BLUE, width=2),
                            hovertemplate="Review top %{x:.1f}% → %{y:.0f}% of later exclusions<extra></extra>")
            fig.update_xaxes(title="Provider-years reviewed, highest risk first (%)", range=[0, 100])
            fig.update_yaxes(title="Later exclusions captured (%)", range=[0, 100])
            st.plotly_chart(fig_layout(fig), width="stretch")
        else:
            st.info("No later-excluded providers in the current filter.")
    with right:
        st.subheader("AUC by group (95% CI)")
        if val:
            rows = [("All (NPI match)", val["primary_npi_match"]),
                    ("Fraud-related types", val["fraud_related_npi_match"]),
                    ("+ name/state matches", val["sensitivity_npi_or_name_state"])]
            rows += [(short(k), v) for k, v in val["by_specialty"].items()]
            fig = go.Figure()
            for name, m in rows[::-1]:
                e = m["auc_risk_score"]
                lo, hi = e.get("ci95", [e["est"], e["est"]])
                fig.add_scatter(x=[lo, hi], y=[name, name], mode="lines", line=dict(color=BLUE, width=3),
                                showlegend=False, hoverinfo="skip")
                fig.add_scatter(x=[e["est"]], y=[name], mode="markers", showlegend=False,
                                marker=dict(color=BLUE, size=11, line=dict(color=SURFACE, width=2)),
                                hovertemplate=f"{name}<br>AUC {e['est']:.2f} [{lo:.2f}–{hi:.2f}]<br>"
                                              f"{m['positive_providers']} excluded providers<extra></extra>")
            fig.add_vline(x=0.5, line=dict(color=INK2, dash="dash", width=1))
            fig.update_xaxes(title="AUC (0.5 = chance)", range=[0.3, 1])
            st.plotly_chart(fig_layout(fig), width="stretch")
            st.caption("Computed on all data; not affected by the sidebar filters.")

# ---------------------------------------------------------------- peer distributions
with tab_dist:
    c1, c2, c3 = st.columns(3)
    spec = c1.selectbox("Specialty", sel_specs or specs, format_func=short)
    year = c2.selectbox("Year", sel_years or years, index=len(sel_years or years) - 1)
    g = df[(df["specialty"] == spec) & (df["year"] == year)]
    avail = [m for m in METRICS if g[m].notna().sum() >= 30]
    metric = c3.selectbox("Metric", avail, format_func=lambda m: METRICS[m])
    col = f"{metric}__adj" if f"{metric}__adj" in g else metric     # volume-adjusted values used in scoring
    g = g.dropna(subset=[col])
    raw = g[col].clip(lower=1e-6)
    med = raw.median()
    mad = 1.4826 * (raw - med).abs().median()
    thr = med + Z_FLAG * mad
    scale = st.radio("Scale", ["Auto", "Linear", "Log"], horizontal=True, key="scale")
    skewed = bool(raw.max() / max(raw.quantile(0.05), 1e-6) > 20)
    use_log = scale == "Log" or (scale == "Auto" and skewed)
    xs = np.log10(raw) if use_log else raw
    fig = go.Figure()
    counts, edges = np.histogram(xs, bins=80)          # bin here: send 80 bars, not every provider
    mids = (edges[:-1] + edges[1:]) / 2
    lo_hi = (10 ** edges[:-1], 10 ** edges[1:]) if use_log else (edges[:-1], edges[1:])
    fig.add_bar(x=mids, y=counts, width=np.diff(edges), name="Peers",
                marker=dict(color=BLUE, line=dict(color=SURFACE, width=0.5)),
                customdata=np.c_[lo_hi], hovertemplate="%{customdata[0]:,.2f}–%{customdata[1]:,.2f}: "
                                                         "%{y:,} providers<extra></extra>")
    ex = g[g["label"] == 1]
    if len(ex):
        ex_x = np.log10(ex[col].clip(lower=1e-6)) if use_log else ex[col]
        fig.add_scatter(x=ex_x, y=np.zeros(len(ex)), mode="markers", name="Later excluded by OIG",
                        marker=dict(color=ORANGE, size=11, line=dict(color=SURFACE, width=2)),
                        hovertemplate="Later excluded<extra></extra>")
    for v, label, color in ((med, "median", INK2), (thr, f"robust z = {Z_FLAG}", ORANGE)):
        fig.add_vline(x=np.log10(v) if use_log else v, line=dict(color=color, width=2 if color == ORANGE else 1))
        fig.add_annotation(x=np.log10(v) if use_log else v, y=1.02, yref="paper", text=label, showarrow=False,
                           xanchor="left", font=dict(size=12, color=INK2))
    if use_log:
        ticks = [t for t in (0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000)
                 if xs.min() <= np.log10(t) <= xs.max()]
        fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[str(t) for t in ticks])
    fig.update_xaxes(title=METRICS[metric] + (", volume-adjusted" if col != metric else "")
                     + (" (log scale)" if use_log else ""))
    fig.update_yaxes(title="Providers")
    st.plotly_chart(fig_layout(fig, 420, bargap=0), width="stretch")
    above = (raw > thr).mean() * 100
    st.caption(f"{len(g):,} providers · median {med:,.2f} · {above:.1f}% above the robust-z threshold "
               f"({thr:,.2f}), vs 0.02% expected if the data were normal: claims metrics are heavily right-skewed.")

# ---------------------------------------------------------------- review queue
with tab_queue:
    c1, c2 = st.columns([1, 3])
    tiers = c1.multiselect("Tier", ["Top 1%", "Top 5%"], default=["Top 1%"])
    q = f[f["tier"].isin(tiers)].sort_values("risk_pct", ascending=False)
    c2.write("")
    c2.caption(f"{len(q):,} provider-years in queue. Most will have legitimate explanations; "
               "the queue sets review priority only.")
    show = ["provider_id", "specialty", "state", "year", "tier", "risk_pctile", "n_flags", "top_drivers"]
    q = q.head(2000)
    if SHOW_IDS:
        show = ["npi", "last_name", "first_name"] + show
    table = q[show].rename(columns={"provider_id": "Provider", "specialty": "Specialty", "state": "State",
                                    "year": "Year", "tier": "Tier", "risk_pctile": "Risk percentile",
                                    "n_flags": "Metrics flagged", "top_drivers": "Top drivers (robust z)"})
    table["Specialty"] = table["Specialty"].map(short)
    st.dataframe(table, width="stretch", hide_index=True, height=320)

    ids = q["provider_id"].drop_duplicates().tolist()
    if ids:
        pid = st.selectbox("Provider detail", ids)
        hist = df[df["provider_id"] == pid].sort_values("year")
        row = q[q["provider_id"] == pid].iloc[0]
        st.markdown(f"**{pid}** · {short(row['specialty'])} · {row['state']} · "
                    f"{int(row['tot_benes']):,} beneficiaries in {row['year']}")
        a, b = st.columns([3, 2])
        with a:
            peers = df[(df["specialty"] == row["specialty"]) & (df["year"] == row["year"])]
            ms = [m for m in METRICS if pd.notna(row.get(f"{m}__pct"))]
            pct = [row[f"{m}__pct"] * 100 for m in ms]
            fig = go.Figure()
            fig.add_bar(x=pct, y=[METRICS[m] for m in ms], orientation="h", marker_color=BLUE,
                        customdata=np.c_[[row[m] for m in ms], [row.get(f"{m}__adj", row[m]) for m in ms],
                                         [peers[m].median() for m in ms], [row[f"{m}__rz"] for m in ms]],
                        hovertemplate="%{y}<br>Percentile %{x:.1f}<br>Observed %{customdata[0]:,.2f}"
                                      " · volume-adjusted %{customdata[1]:,.2f}<br>Peer median %{customdata[2]:,.2f}"
                                      " · robust z %{customdata[3]:.1f}<extra></extra>")
            fig.add_vline(x=95, line=dict(color=ORANGE, width=1.5, dash="dot"))
            fig.update_xaxes(title=f"Percentile vs {short(row['specialty'])} peers, {row['year']}", range=[0, 100])
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_layout(fig, 60 + 32 * len(ms), bargap=0.35), width="stretch")
        with b:
            fig = go.Figure()
            fig.add_scatter(x=hist["year"], y=hist["risk_pct"] * 100, mode="lines+markers",
                            line=dict(color=BLUE, width=2), marker=dict(size=9, line=dict(color=SURFACE, width=2)),
                            hovertemplate="%{x}: risk percentile %{y:.1f}<extra></extra>", name="Risk percentile")
            fig.add_hline(y=95, line=dict(color=ORANGE, width=1.5, dash="dot"))
            fig.update_xaxes(title="Data year", dtick=1)
            fig.update_yaxes(title="Risk percentile", range=[0, 100])
            st.plotly_chart(fig_layout(fig, 300, showlegend=False), width="stretch")
            st.caption("Dotted line = top-5% threshold.")

# ---------------------------------------------------------------- method
with tab_method:
    st.markdown(f"""
**Question.** Can peer-benchmarked outlier scores from public CMS data rank providers so that those later
excluded by HHS-OIG are concentrated at the top?

**Data.** CMS Medicare Physician & Other Practitioners (by provider, and by provider and service) and
Part D Prescribers (by provider), 2016–2024; HHS-OIG LEIE. Scope: Interventional Pain Management,
Pain Management, Physical Therapist in Private Practice.

**Scoring.** Peer group = specialty × year. Rates from low-volume providers are first shrunk toward the
peer median (empirical-Bayes credibility weighting), so a provider with 15 patients can't top the list on
noise alone. Each metric then gets a robust z-score, (x − median) / (1.4826 × MAD),
because claims metrics are heavily right-skewed. Rule score = mean of a provider's three largest positive
robust z's (a single metric is flagged above {Z_FLAG}). An Isolation Forest per peer group captures unusual
combinations. Risk score = 2:1 weighted average of the two within-group percentiles.

**Validation.** Label = exclusion of the same NPI within 3 years after the data year. Confidence intervals come
from a bootstrap that resamples providers. Name + state matches are a sensitivity check only.

**Caveats.**
- An outlier isn't evidence of fraud. Case mix, subspecialty and referral patterns can explain high values.
- OIG exclusion lags misconduct by years and covers only cases OIG acted on. Reinstated providers come from archived LEIE snapshots and OIG's monthly supplements.
- Positive providers are few (67), so intervals are wide.
- Cells with fewer than 11 beneficiaries are suppressed by CMS; small providers aren't scored.
""")
