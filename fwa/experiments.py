"""Experiments compared against a frozen baseline (see README, "Experiments").

  baseline  the production risk score, frozen with `fwa experiments --freeze`
  A         supervised XGBoost on the same metrics + case-mix, NPI-grouped CV (A1) and
            NPI-grouped + time split: train 2021-22, test 2023-24 (A2)
  B         case-mix peers: each metric residualized on patient mix (age, dual eligibility,
            chronic conditions, risk score) inside specialty x year, then the same rule score
  C         other anomaly detectors in place of Isolation Forest (ECOD, CBLOF; PyOD)
  D         replication ladder of Johnson & Khoshgoftaar (2023): their label and features, then
            one change at a time toward this project's evaluation

Everything is decided here before results are seen; results go to reports/<run>/experiments.md.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from .config import Config
from .evaluate import compare, evaluate, fmt
from .load import connect
from .score import METRICS, _robust_z
from .validate import labelled

log = logging.getLogger(__name__)
SEED = 42
PAPER_TYPES = ["1128a1", "1128a2", "1128a3", "1128b4", "1128b7"]   # their Table 6 (c)(3)(g) codes rare


# ------------------------------------------------------------------ data
def _dir(cfg: Config):
    d = cfg.data_dir / "experiments"
    d.mkdir(exist_ok=True)
    return d


def _cols(con, table: str) -> set[str]:
    return {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}


def features(cfg: Config) -> pd.DataFrame:
    """Scored provider-years + case-mix covariates from the CMS by-provider file (columns that a
    release lacks come back NULL)."""
    con = connect(cfg)
    have = _cols(con, "raw.physician_provider")
    v = lambda c: f"TRY_CAST({c} AS DOUBLE)" if c in have else "NULL::DOUBLE"   # noqa: E731
    benes = f"NULLIF({v('Tot_Benes')}, 0)"
    ccs = [c for c in ("Bene_CC_PH_Diabetes_V2_Pct", "Bene_CC_PH_Arthritis_V2_Pct", "Bene_CC_PH_Hypertension_V2_Pct",
                       "Bene_CC_PH_CKD_V2_Pct", "Bene_CC_BH_Depress_V1_Pct", "Bene_CC_PH_Osteoporosis_V2_Pct") if c in have]
    chronic = ("(" + " + ".join(f"COALESCE({v(c)}, 0)" for c in ccs) + f") / {len(ccs)}") if ccs else "NULL::DOUBLE"
    df = con.execute(f"""
        WITH cm AS (
            SELECT CAST(year AS INTEGER) AS year, Rndrng_NPI AS npi,
                   {v('Bene_Avg_Age')}                        AS cm_avg_age,
                   {v('Bene_Age_GT_84_Cnt')} / {benes}        AS cm_share_85plus,
                   {v('Bene_Dual_Cnt')} / {benes}             AS cm_dual_share,
                   {v('Bene_Feml_Cnt')} / {benes}             AS cm_female_share,
                   {v('Bene_Avg_Risk_Scre')}                  AS cm_risk_score,
                   {chronic}                                  AS cm_chronic_pct
            FROM raw.physician_provider
        )
        SELECT s.*, cm.* EXCLUDE (year, npi)
        FROM mart.provider_scores s LEFT JOIN cm USING (year, npi)
    """).df()
    con.close()
    return df


CASEMIX = ["cm_avg_age", "cm_share_85plus", "cm_dual_share", "cm_female_share", "cm_risk_score", "cm_chronic_pct"]


def _labels(cfg: Config) -> pd.DataFrame:
    return labelled(cfg)


# ------------------------------------------------------------------ baseline
def freeze(cfg: Config) -> None:
    con = connect(cfg)
    b = con.execute("SELECT year, npi, specialty, risk_score, composite_pct, iforest_pct FROM mart.provider_scores").df()
    con.close()
    b.to_parquet(_dir(cfg) / "baseline_scores.parquet", index=False)
    meta = {"frozen_at": time.strftime("%Y-%m-%d %H:%M"), "rows": len(b), "scoring": cfg["scoring"]}
    (_dir(cfg) / "baseline_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("baseline frozen: %d rows", len(b))


def baseline_scores(cfg: Config) -> pd.DataFrame:
    p = _dir(cfg) / "baseline_scores.parquet"
    if not p.exists():
        freeze(cfg)
    return pd.read_parquet(p)


# ------------------------------------------------------------------ A: supervised XGBoost
def _xy(df: pd.DataFrame):
    cols = [f"{m}__adj" for m in METRICS if f"{m}__adj" in df] + CASEMIX + ["tot_benes"]
    X = df[cols].astype(float).copy()
    X["tot_benes"] = np.log1p(X["tot_benes"])
    X = pd.concat([X, pd.get_dummies(df["specialty"], prefix="spec", dtype=float)], axis=1)
    return X, cols


def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=5, eval_metric="aucpr", tree_method="hist", random_state=SEED, n_jobs=-1)


def exp_a(cfg: Config, lab: pd.DataFrame) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold
    df = features(cfg).merge(lab[["year", "npi", "label"]], on=["year", "npi"])
    X, _ = _xy(df)
    y, groups = df["label"].to_numpy(), df["npi"].to_numpy()
    out = {}
    # A1: 5-fold CV grouped by NPI (a provider is never in train and test)
    oof = np.full(len(df), np.nan)
    contrib = np.zeros(X.shape[1])
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups):
        m = _xgb().set_params(scale_pos_weight=(y[tr] == 0).sum() / max(y[tr].sum(), 1))
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        contrib += np.abs(m.get_booster().predict(__import__("xgboost").DMatrix(X.iloc[te]), pred_contribs=True)[:, :-1]).mean(0)
    df["a1"] = oof
    top = pd.Series(contrib / 5, index=X.columns).sort_values(ascending=False).head(8)
    out["A1"] = {"scores": df[["year", "npi", "a1"]].rename(columns={"a1": "score"}),
                 "note": "XGBoost, 5-fold CV grouped by NPI", "top_features": top.round(4).to_dict()}
    # A2: grouped AND temporal - train on 2021-22 rows of training NPIs, score 2023-24 rows of held-out NPIs
    yrs = sorted(df["year"].unique())
    cut = yrs[len(yrs) // 2 - 1]
    oof2 = np.full(len(df), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups):
        trm = np.zeros(len(df), bool)
        trm[tr] = True
        tr_idx = np.where(trm & (df["year"].to_numpy() <= cut))[0]
        te_idx = np.where(~trm & (df["year"].to_numpy() > cut))[0]
        if y[tr_idx].sum() == 0:
            continue
        m = _xgb().set_params(scale_pos_weight=(y[tr_idx] == 0).sum() / y[tr_idx].sum())
        m.fit(X.iloc[tr_idx], y[tr_idx])
        oof2[te_idx] = m.predict_proba(X.iloc[te_idx])[:, 1]
    df["a2"] = oof2
    out["A2"] = {"scores": df[["year", "npi", "a2"]].dropna().rename(columns={"a2": "score"}),
                 "note": f"XGBoost, NPI-grouped + trained on years <= {cut}, tested on later years (test rows only)"}
    return out


# ------------------------------------------------------------------ B: case-mix peers
def _rule_score(z: pd.DataFrame, k: int = 3) -> pd.Series:
    rz = z.clip(lower=0, upper=10)
    return rz.apply(lambda r: np.sort(r.dropna().values)[::-1][:k].mean() if r.notna().any() else 0.0, axis=1)


def exp_b(cfg: Config, lab: pd.DataFrame) -> dict:
    df = features(cfg)
    parts = []
    for _, g in df.groupby(["year", "specialty"]):
        g = g.copy()
        C = g[CASEMIX].astype(float)
        C = (C - C.mean()) / C.std(ddof=0).replace(0, 1)
        C = np.c_[np.ones(len(g)), C.fillna(0).to_numpy()]
        zs = {}
        for m in METRICS:
            col = f"{m}__adj"
            if col not in g or g[col].notna().sum() < 30:
                continue
            yv = np.log1p(g[col].clip(lower=0).astype(float))
            ok = yv.notna().to_numpy()
            beta, *_ = np.linalg.lstsq(C[ok], yv[ok].to_numpy(), rcond=None)
            resid = pd.Series(np.nan, index=g.index)
            resid[ok] = yv[ok].to_numpy() - C[ok] @ beta
            zs[m] = _robust_z(resid)
        g["b"] = _rule_score(pd.DataFrame(zs))
        parts.append(g[["year", "npi", "b"]])
    s = pd.concat(parts).rename(columns={"b": "score"})
    return {"B": {"scores": s, "note": "rule score on metrics residualized for patient mix (log scale, OLS per specialty-year)"}}


# ------------------------------------------------------------------ C: other detectors
def exp_c(cfg: Config, lab: pd.DataFrame) -> dict:
    from pyod.models.cblof import CBLOF
    from pyod.models.ecod import ECOD
    from sklearn.preprocessing import StandardScaler
    df = features(cfg)
    sc = cfg["scoring"]
    res = {"ecod": [], "cblof": []}
    for _, g in df.groupby(["year", "specialty"]):
        ms = [m for m in METRICS if f"{m}__adj" in g and g[f"{m}__adj"].notna().mean() >= sc["iforest_min_coverage"]]
        X = g[[f"{m}__adj" for m in ms]].apply(lambda c: np.log1p(c.clip(lower=0))).apply(lambda c: c.fillna(c.median()))
        X = StandardScaler().fit_transform(X)
        for name, model in (("ecod", ECOD()), ("cblof", CBLOF(n_clusters=8, random_state=SEED))):
            model.fit(X)
            res[name].append(pd.DataFrame({"year": g["year"].to_numpy(), "npi": g["npi"].to_numpy(),
                                           "det": pd.Series(model.decision_scores_).rank(pct=True).to_numpy(),
                                           "comp": g["composite_pct"].to_numpy()}))
    w = sc["weights"]
    out = {}
    for name, parts in res.items():
        d = pd.concat(parts)
        out[f"C_{name}"] = {"scores": d.assign(score=(w["composite"] * d["comp"] + w["iforest"] * d["det"]) /
                                              (w["composite"] + w["iforest"]))[["year", "npi", "score"]],
                            "note": f"baseline rules + {name.upper()} in place of Isolation Forest (2:1)"}
        out[f"C_{name}_alone"] = {"scores": d.rename(columns={"det": "score"})[["year", "npi", "score"]],
                                  "note": f"{name.upper()} alone"}
    return out


# ------------------------------------------------------------------ D: replication ladder
def _paper_features(cfg: Config) -> pd.DataFrame:
    """'Aggregated-enriched': 6 summary stats over each provider's code lines + by-provider totals and
    beneficiary features (their Table 4 / enrichment), all specialties in scope pooled."""
    con = connect(cfg)
    have_s, have_p = _cols(con, "raw.physician_service"), _cols(con, "raw.physician_provider")
    agg = []
    for c in [c for c in ("Tot_Srvcs", "Tot_Benes", "Tot_Bene_Day_Srvcs", "Avg_Sbmtd_Chrg", "Avg_Mdcr_Pymt_Amt") if c in have_s]:
        v = f"TRY_CAST({c} AS DOUBLE)"
        agg += [f"min({v}) AS {c}_min", f"max({v}) AS {c}_max", f"median({v}) AS {c}_median",
                f"avg({v}) AS {c}_mean", f"sum({v}) AS {c}_sum", f"coalesce(stddev_pop({v}), 0) AS {c}_std"]
    num = ["Tot_HCPCS_Cds", "Tot_Benes", "Tot_Srvcs", "Tot_Sbmtd_Chrg", "Tot_Mdcr_Pymt_Amt", "Tot_Mdcr_Stdzd_Amt",
           "Bene_Avg_Age", "Bene_Age_LT_65_Cnt", "Bene_Age_65_74_Cnt", "Bene_Age_75_84_Cnt", "Bene_Age_GT_84_Cnt",
           "Bene_Feml_Cnt", "Bene_Male_Cnt", "Bene_Dual_Cnt", "Bene_Ndual_Cnt", "Bene_Avg_Risk_Scre"]
    num = [c for c in num if c in have_p]
    cc = sorted(c for c in have_p if c.startswith("Bene_CC_"))
    sbp = ", ".join(f"COALESCE(TRY_CAST(p.{c} AS DOUBLE), 0) AS {c}" for c in num + cc)
    df = con.execute(f"""
        WITH a AS (SELECT CAST(year AS INTEGER) AS year, Rndrng_NPI AS npi, {', '.join(agg)}
                   FROM raw.physician_service GROUP BY ALL)
        SELECT a.*, p.Rndrng_Prvdr_Type AS specialty, {sbp}
        FROM a JOIN raw.physician_provider p ON CAST(p.year AS INTEGER) = a.year AND p.Rndrng_NPI = a.npi
        WHERE TRY_CAST(p.Tot_Benes AS DOUBLE) >= 11
    """).df()
    leie = con.execute("SELECT npi, excl_type, excl_date FROM stg.leie WHERE npi IS NOT NULL").df()
    con.close()
    # their label: provider in LEIE under a fraud-related rule; years up to exclusion year + 5 are positive
    leie = leie[leie["excl_type"].isin(PAPER_TYPES)]
    last = (pd.to_datetime(leie["excl_date"]).dt.year + 5).groupby(leie["npi"]).max()
    df["paper_label"] = (df["year"] <= df["npi"].map(last)).fillna(False).astype(int)
    return df


def exp_d(cfg: Config, lab: pd.DataFrame) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    df = _paper_features(cfg)
    Xcols = [c for c in df.columns if c not in ("year", "npi", "specialty", "paper_label")]
    X = pd.concat([df[Xcols].astype(float), pd.get_dummies(df["specialty"], prefix="spec", dtype=float)], axis=1)
    y, groups = df["paper_label"].to_numpy(), df["npi"].to_numpy()

    def oof(splitter, yv, use_groups):
        p = np.full(len(df), np.nan)
        splits = splitter.split(X, yv, groups) if use_groups else splitter.split(X, yv)
        for tr, te in splits:
            m = _xgb().set_params(scale_pos_weight=(yv[tr] == 0).sum() / max(yv[tr].sum(), 1))
            m.fit(X.iloc[tr], yv[tr])
            p[te] = m.predict_proba(X.iloc[te])[:, 1]
        return p

    df["d1"] = oof(StratifiedKFold(5, shuffle=True, random_state=SEED), y, False)
    df["d2"] = oof(StratifiedGroupKFold(5, shuffle=True, random_state=SEED), y, True)
    # D4: our prospective label with their features (rows the labeller kept)
    d4 = df.merge(lab[["year", "npi", "label"]], on=["year", "npi"])
    y4 = d4["label"].to_numpy()
    X4 = pd.concat([d4[Xcols].astype(float), pd.get_dummies(d4["specialty"], prefix="spec", dtype=float)], axis=1)
    p4 = np.full(len(d4), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X4, y4, d4["npi"]):
        m = _xgb().set_params(scale_pos_weight=(y4[tr] == 0).sum() / max(y4[tr].sum(), 1))
        m.fit(X4.iloc[tr], y4[tr])
        p4[te] = m.predict_proba(X4.iloc[te])[:, 1]
    d4["d4"] = p4
    plab = df[["year", "npi", "specialty", "paper_label"]].assign(tot_stdzd_pymt=df["Tot_Mdcr_Stdzd_Amt"])
    return {"_ladder": {"df": df, "d4": d4, "plab": plab}}


# ------------------------------------------------------------------ runner
# Pre-registered (Sep 30, 2026, before any 2016-2020 data was downloaded): CBLOF in place of Isolation
# Forest is adopted only if, on the held-out data years 2016-2019, the paired bootstrap difference
# (CBLOF combined minus baseline) has an AUC interval above 0, or a top-5% dollar-recall interval above 0
# with an AUC difference >= 0. 1,000 provider resamples.
CONFIRM_YEARS = (2016, 2019)


def run(cfg: Config, only: list[str] | None = None, freeze_baseline: bool = False, reps: int = 300,
        years: tuple[int, int] | None = None) -> None:
    if freeze_baseline:
        freeze(cfg)
    lab = _labels(cfg)
    if years:
        lab = lab[lab["year"].between(*years)]
        log.info("evaluating data years %s-%s only (%d provider-years)", years[0], years[1], len(lab))
    results_path = _dir(cfg) / ("results.json" if not years else f"results_{years[0]}_{years[1]}.json")
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    todo = only or ["baseline", "A", "B", "C", "D", "confirm"]
    runners = {"A": exp_a, "B": exp_b, "C": exp_c, "D": exp_d}
    for name in todo:
        t0 = time.time()
        if name == "confirm":
            full = _labels(cfg)
            held = full[full["year"].between(*CONFIRM_YEARS)]
            if held.empty:
                log.warning("confirmatory test skipped: no data years %s-%s", *CONFIRM_YEARS)
                continue
            base = baseline_scores(cfg).rename(columns={"risk_score": "score"})
            cb = exp_c(cfg, held)["C_cblof"]["scores"]
            res = compare(base, cb, held, reps=1000)
            a, dd = res["auc_diff"], res["dollar_recall_diff"]
            res["adopt"] = bool(a["ci95"][0] > 0 or (dd["ci95"][0] > 0 and a["est"] >= 0))
            results["confirm_cblof"] = res
        elif name == "baseline":
            b = baseline_scores(cfg)
            for col, note in (("risk_score", "production score (rules 2 : IF 1)"),
                              ("composite_pct", "rules only"), ("iforest_pct", "Isolation Forest only")):
                results[f"baseline_{col}"] = {"note": note, **evaluate(b, lab, col, reps=reps)}
        elif name == "D":
            L = exp_d(cfg, lab)["_ladder"]
            df, d4, plab = L["df"], L["d4"], L["plab"]
            steps = [("D1", "their label, all specialties pooled, row-level CV, pooled AUC", df, "d1", plab, "paper_label", False),
                     ("D2", "their label, NPI-grouped CV, pooled AUC", df, "d2", plab, "paper_label", False),
                     ("D3", "their label, NPI-grouped CV, AUC within specialty-year", df, "d2", plab, "paper_label", True),
                     ("D4", "our label (future exclusion), NPI-grouped CV, within specialty-year", d4, "d4", lab, "label", True)]
            for key, note, frame, col, labels, lcol, within in steps:
                sc = frame[["year", "npi", col]].rename(columns={col: "score"})
                results[key] = {"note": note, **evaluate(sc, labels, "score", lcol, within_group=within, reps=reps)}
        else:
            for key, r in runners[name](cfg, lab).items():
                results[key] = {"note": r["note"], **evaluate(r["scores"], lab, "score", reps=reps)}
                if "top_features" in r:
                    results[key]["top_features"] = r["top_features"]
        log.info("experiment %s done in %.0fs", name, time.time() - t0)
        results_path.write_text(json.dumps(results, indent=2, default=float))
    report(cfg, results)


def report(cfg: Config, results: dict) -> None:
    order = ["baseline_risk_score", "baseline_composite_pct", "baseline_iforest_pct", "A1", "A2", "B",
             "C_ecod", "C_cblof", "C_ecod_alone", "C_cblof_alone"]
    lines = ["# Experiments vs frozen baseline", "",
             "Label: same-NPI OIG exclusion within the horizon after the data year (cumulative LEIE when enabled). "
             "Scores ranked within specialty × year. 95% CIs resample providers.", "",
             "| Run | What | AUC | Top 5% lift | Top 5% recall | Top 5% $ recall |", "|---|---|---|---|---|---|"]
    for k in order:
        if k in results:
            r = results[k]
            lines.append(f"| {k} | {r['note']} | {fmt(r['auc'])} | {fmt(r['top5_lift'], 1)} | "
                         f"{fmt(r['top5_recall'], pct=True)} | {fmt(r['top5_dollar_recall'], pct=True)} |")
    if any(k in results for k in ("D1", "D2", "D3", "D4")):
        lines += ["", "## Replication ladder (Johnson & Khoshgoftaar 2023 setup → this project's)", "",
                  "| Step | Setup | Positive providers | AUC |", "|---|---|---|---|"]
        for k in ("D1", "D2", "D3", "D4"):
            if k in results:
                r = results[k]
                lines.append(f"| {k} | {r['note']} | {r['positive_providers']} | {fmt(r['auc'])} |")
    if "confirm_cblof" in results:
        c = results["confirm_cblof"]
        lines += ["", f"## Pre-registered test: CBLOF vs Isolation Forest on held-out years {CONFIRM_YEARS[0]}-{CONFIRM_YEARS[1]}", "",
                  f"{c['positive_providers']} later-excluded providers. Paired bootstrap (1,000 provider resamples), CBLOF minus baseline:", "",
                  f"- AUC difference: {fmt(c['auc_diff'], 3)}",
                  f"- Top-5% dollar-recall difference: {fmt(c['dollar_recall_diff'], 3)}",
                  f"- Decision: **{'adopt CBLOF' if c['adopt'] else 'keep Isolation Forest'}**"]
    if "A1" in results and "top_features" in results["A1"]:
        lines += ["", "A1 top features (mean |SHAP|): " + ", ".join(f"{k} {v:.3f}" for k, v in results["A1"]["top_features"].items())]
    (cfg.reports_dir / "experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("report: %s", cfg.reports_dir / "experiments.md")
