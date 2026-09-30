from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


@dataclass
class Config:
    raw: dict
    synthetic: bool = False

    @classmethod
    def load(cls, path: Path | None = None, synthetic: bool = False) -> "Config":
        with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        local = ROOT / "config.local.yaml"            # machine-specific overrides, git-ignored
        if path is None and local.exists():
            with open(local, encoding="utf-8") as f:
                for k, v in (yaml.safe_load(f) or {}).items():
                    raw[k] = {**raw.get(k, {}), **v} if isinstance(v, dict) and isinstance(raw.get(k), dict) else v
        return cls(raw, synthetic)

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def data_dir(self) -> Path:
        d = ROOT / ("data_synthetic" if self.synthetic else self.raw["data_dir"])
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def out_dir(self) -> Path:
        d = self.data_dir / "out"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def db_path(self) -> Path:
        # FWA_DB_PATH lets you keep the database somewhere else (e.g. a faster local disk)
        override = os.environ.get("FWA_DB_PATH")
        return Path(override) if override else self.data_dir / self.raw["db_file"]

    @property
    def reports_dir(self) -> Path:
        d = ROOT / "reports" / ("synthetic" if self.synthetic else "real")
        d.mkdir(parents=True, exist_ok=True)
        return d
