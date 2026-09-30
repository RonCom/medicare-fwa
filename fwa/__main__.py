"""CLI:  python -m fwa [download|load|transform|score|validate|charts|audit|all] [--synthetic] [--force]"""
from __future__ import annotations

import argparse
import logging

from . import charts, download, load, score, synthetic, validate
from .config import Config


def main() -> None:
    ap = argparse.ArgumentParser(prog="fwa")
    ap.add_argument("step", choices=["download", "load", "transform", "score", "validate", "charts", "experiments", "audit", "audit-bandit", "all",
                                       "sf-keygen", "sf-load", "sf-transform", "sf-score", "sf-reconcile", "sf-all"])
    ap.add_argument("--synthetic", action="store_true", help="use generated data in data_synthetic/")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--only", help="experiments to run, comma-separated: baseline,A,B,C,D")
    ap.add_argument("--freeze", action="store_true", help="freeze the current scores as the experiment baseline")
    ap.add_argument("--reps", type=int, default=300, help="bootstrap reps for experiments")
    ap.add_argument("--years", help="evaluate experiments on a range of data years, e.g. 2021-2024")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.load(synthetic=a.synthetic)

    steps = {"all": ["download", "load", "transform", "score", "validate", "charts"],
             "sf-all": ["sf-load", "sf-transform", "sf-score", "sf-reconcile"]}.get(a.step, [a.step])
    for s in steps:
        if s == "download":
            synthetic.run(cfg) if a.synthetic else download.run(cfg, a.force)
        elif s == "load":
            load.load_raw(cfg)
        elif s == "transform":
            load.transform(cfg)
        elif s == "score":
            score.run(cfg)
        elif s == "validate":
            validate.run(cfg)
        elif s == "charts":
            charts.run(cfg)
        elif s.startswith("sf-"):
            from . import snowflake as sf
            if s == "sf-keygen":
                sf.keygen(cfg, a.force)
            elif s == "sf-load":
                sf.load_raw(cfg)
            elif s == "sf-transform":
                load.transform(cfg, target="snowflake")
            elif s == "sf-score":
                sf.score(cfg)
            elif s == "sf-reconcile":
                sf.reconcile(cfg)
        elif s == "audit":
            from . import audit
            audit.run(cfg)
        elif s == "audit-bandit":
            from . import audit_bandit
            audit_bandit.run(cfg)
        elif s == "experiments":
            from . import experiments
            yrs = tuple(int(y) for y in a.years.split("-")) if a.years else None
            experiments.run(cfg, a.only.split(",") if a.only else None, a.freeze, a.reps, yrs)


if __name__ == "__main__":
    main()
