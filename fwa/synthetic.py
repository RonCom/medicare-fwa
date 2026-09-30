"""Generate schema-faithful synthetic CMS + LEIE files so the pipeline runs offline.

~2% of providers are planted anomalies (inflated utilization, upcoding, heavy opioid prescribing)
and are far more likely to appear on the synthetic exclusion list. Numbers are NOT realistic
estimates of real Medicare data; they only exercise the code paths.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config, slug

log = logging.getLogger(__name__)

STATES = ["PA", "NJ", "NY", "FL", "TX", "CA", "OH", "NC", "GA", "MI"]
LAST = ["SMITH", "JOHNSON", "LEE", "PATEL", "GARCIA", "BROWN", "NGUYEN", "KIM", "MILLER", "DAVIS", "SHAH", "CHEN"]
FIRST = ["JAMES", "MARY", "ROBERT", "PRIYA", "DAVID", "LINDA", "WEI", "MARIA", "JOHN", "ANITA", "SARAH", "OMAR"]
PAIN_CODES = {"99212": 60, "99213": 85, "99214": 125, "99215": 175, "64483": 210, "64493": 180, "20552": 45, "62323": 230}
PT_CODES = {"97110": 30, "97140": 27, "97530": 35, "97112": 32, "97161": 95, "97035": 12}
SIZES = {"Interventional Pain Management": 1500, "Pain Management": 700, "Physical Therapist in Private Practice": 3000}
YEARS = [2020, 2021, 2022, 2023]


def _providers(rng, specialty, n, start_npi):
    p = pd.DataFrame({
        "npi": [str(start_npi + i) for i in range(n)],
        "last": rng.choice(LAST, n), "first": rng.choice(FIRST, n) + rng.integers(1, 999, n).astype(str),
        "state": rng.choice(STATES, n), "specialty": specialty,
        "anomaly": rng.random(n) < 0.02,
        "size": rng.lognormal(5.0, 0.6, n),           # beneficiaries
        "intensity": rng.lognormal(0, 0.15, n),
        "risk": rng.normal(1.3, 0.2, n).clip(0.6, 3),
    })
    return p


def run(cfg: Config) -> None:
    rng = np.random.default_rng(7)
    raw = cfg.raw_dir
    all_prov = []
    for i, spec in enumerate(cfg["specialties"]):
        all_prov.append(_providers(rng, spec, SIZES.get(spec, 1000), 1_000_000_000 + i * 100_000))
    prov = pd.concat(all_prov, ignore_index=True)

    # exclusion dates (planted anomalies much more likely)
    p_excl = np.where(prov["anomaly"], 0.25, 0.004)
    excluded = rng.random(len(prov)) < p_excl
    days = rng.integers(0, 365 * 5, len(prov))
    prov["excl_date"] = pd.NaT
    prov.loc[excluded, "excl_date"] = pd.Timestamp("2021-06-01") + pd.to_timedelta(days[excluded], unit="D")

    for year in YEARS:
        active = prov[prov["excl_date"].isna() | (prov["excl_date"] > pd.Timestamp(f"{year}-12-31"))]
        for spec, g in active.groupby("specialty"):
            codes = PT_CODES if "Physical Therapist" in spec else PAIN_CODES
            svc_rows, prov_rows, rx_rows = [], [], []
            for r in g.itertuples():
                benes = max(11, int(r.size * rng.lognormal(0, 0.1)))
                boost = 2.2 if r.anomaly else 1.0
                tot_srv = tot_pay = 0.0
                for cd, price in codes.items():
                    if rng.random() < 0.15:
                        continue
                    cb = max(11, int(benes * rng.uniform(0.2, 0.9)))
                    spb = rng.lognormal(0.6, 0.3) * r.intensity
                    if cd in ("99214", "99215") and r.anomaly:
                        spb *= 2.5
                    elif r.anomaly:
                        spb *= boost
                    srv = round(cb * spb)
                    pay = price * rng.uniform(0.95, 1.05)
                    svc_rows.append({"Rndrng_NPI": r.npi, "Rndrng_Prvdr_Last_Org_Name": r.last,
                                     "Rndrng_Prvdr_First_Name": r.first, "Rndrng_Prvdr_State_Abrvtn": r.state,
                                     "Rndrng_Prvdr_Type": spec, "HCPCS_Cd": cd, "Place_Of_Srvc": "O",
                                     "Tot_Benes": cb, "Tot_Srvcs": srv, "Avg_Sbmtd_Chrg": round(pay * rng.uniform(1.5, 3), 2),
                                     "Tot_Bene_Day_Srvcs": max(cb, round(srv / (rng.uniform(1.8, 2.6) if r.anomaly else rng.uniform(1.1, 1.6)))), "Avg_Mdcr_Pymt_Amt": round(pay * 0.8, 2),
                                     "Avg_Mdcr_Stdzd_Amt": round(pay * 0.78, 2)})
                    tot_srv += srv
                    tot_pay += srv * pay
                prov_rows.append({"Rndrng_NPI": r.npi, "Rndrng_Prvdr_Last_Org_Name": r.last,
                                  "Rndrng_Prvdr_First_Name": r.first, "Rndrng_Prvdr_State_Abrvtn": r.state,
                                  "Rndrng_Prvdr_Type": spec, "Tot_HCPCS_Cds": len(codes), "Tot_Benes": benes,
                                  "Tot_Srvcs": tot_srv, "Tot_Mdcr_Pymt_Amt": round(tot_pay * 0.8, 2),
                                  "Tot_Mdcr_Stdzd_Amt": round(tot_pay * 0.78, 2),
                                  "Bene_Avg_Risk_Scre": round(r.risk, 4),
                                  "Bene_Avg_Age": round(rng.normal(72, 4), 1),
                                  "Bene_Age_GT_84_Cnt": int(benes * rng.uniform(0.05, 0.2)),
                                  "Bene_Dual_Cnt": int(benes * rng.uniform(0.05, 0.4)),
                                  "Bene_Feml_Cnt": int(benes * rng.uniform(0.45, 0.7)),
                                  "Bene_CC_PH_Diabetes_V2_Pct": round(rng.uniform(15, 40), 1),
                                  "Bene_CC_PH_Arthritis_V2_Pct": round(rng.uniform(30, 70), 1)})
                if "Physical Therapist" not in spec:
                    clms = int(benes * rng.uniform(3, 8))
                    op = rng.beta(2, 8) * (2.5 if r.anomaly else 1)
                    rx_rows.append({"Prscrbr_NPI": r.npi, "Prscrbr_Last_Org_Name": r.last,
                                    "Prscrbr_First_Name": r.first, "Prscrbr_State_Abrvtn": r.state,
                                    "Prscrbr_Type": spec, "Tot_Clms": clms, "Tot_Benes": benes,
                                    "Tot_Drug_Cst": round(clms * rng.lognormal(3.5, 0.4), 2),
                                    "Tot_Day_Suply": clms * int(rng.integers(20, 35)),
                                    "Brnd_Tot_Clms": int(clms * rng.beta(2, 6)),
                                    "Opioid_Tot_Clms": int(clms * min(op, 0.95)),
                                    "Opioid_LA_Tot_Clms": int(clms * min(op, 0.95) * rng.beta(2, 8) * (2 if r.anomaly else 1))})
            for name, rows in (("physician_service", svc_rows), ("physician_provider", prov_rows), ("partd_provider", rx_rows)):
                if not rows:
                    continue
                dest = raw / name / f"year={year}" / f"{slug(spec)}.parquet"
                dest.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).astype("string").to_parquet(dest, index=False)

    # LEIE: planted exclusions (40% without NPI to exercise name matching) + unrelated noise rows
    ex = prov[prov["excl_date"].notna()]
    no_npi = rng.random(len(ex)) < 0.4
    leie = pd.DataFrame({
        "LASTNAME": ex["last"], "FIRSTNAME": ex["first"], "MIDNAME": "", "BUSNAME": "", "GENERAL": "INDIVIDUAL",
        "SPECIALTY": ex["specialty"].str.upper(), "UPIN": "", "NPI": np.where(no_npi, "0000000000", ex["npi"]),
        "DOB": "", "ADDRESS": "", "CITY": "", "STATE": ex["state"], "ZIP": "",
        "EXCLTYPE": rng.choice(["1128a1", "1128a3", "1128b4", "1128b7"], len(ex)),
        "EXCLDATE": ex["excl_date"].dt.strftime("%Y%m%d"), "REINDATE": "00000000",
        "WAIVERDATE": "00000000", "WVRSTATE": ""})
    noise = leie.sample(n=min(500, len(leie)), replace=True, random_state=1).assign(
        NPI="0000000000", LASTNAME="OTHER", FIRSTNAME="PERSON", EXCLDATE="20150101")
    # ~30% of exclusions are later reinstated: they vanish from the current list but stay in
    # the yearly snapshots taken while they were active (what the cumulative history recovers)
    excl_dt = pd.to_datetime(leie["EXCLDATE"])
    rein = rng.random(len(leie)) < 0.3
    rein_dt = excl_dt + pd.to_timedelta(rng.integers(365, 900, len(leie)), unit="D")
    rein_dt = rein_dt.where(rein & (rein_dt < pd.Timestamp("2026-09-01")))
    full = pd.concat([leie, noise])
    current = pd.concat([leie[rein_dt.isna().to_numpy()], noise])
    dest = raw / "leie" / "UPDATED.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(dest, index=False)
    hist = raw / "leie" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    for y in range(2020, 2027):
        snap = pd.Timestamp(f"{y}-01-05")
        active = (excl_dt <= snap) & (rein_dt.isna() | (rein_dt > snap))
        pd.concat([leie[active.to_numpy()], noise]).to_csv(hist / f"UPDATED_{y}0105000000.csv", index=False)
    log.info("synthetic: %d providers, %d planted anomalies, %d exclusions",
             len(prov), prov["anomaly"].sum(), len(ex))
    prov[["npi", "specialty", "anomaly"]].to_parquet(cfg.out_dir / "synthetic_truth.parquet", index=False)
    _synthetic_mue(cfg, rng)


def _synthetic_mue(cfg: Config, rng) -> None:
    """Zip files shaped like the CMS release: disclaimer lines, then a header, then rows."""
    import io
    import zipfile
    codes = {**PAIN_CODES, **PT_CODES}
    for year in YEARS:
        lines = ["CPT codes, descriptions and other data only are copyright AMA. All rights reserved.",
                 "Practitioner Services MUE Table (synthetic)", "",
                 '"HCPCS/CPT Code","Practitioner Services MUE Values","MUE Adjudication Indicator","MUE Rationale"']
        for cd in codes:
            mue = 1 if cd.startswith("992") or cd in ("97161", "64493") else int(rng.integers(2, 6))
            lines.append(f'"{cd}","{mue}","3 Date of Service Edit: Clinical","Clinical: Data"')
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(f"MCR_MUE_PractitionerServices_Eff_10-01-{year}.csv", "\n".join(lines))
        dest = cfg.raw_dir / "mue" / f"{year}.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(buf.getvalue())
