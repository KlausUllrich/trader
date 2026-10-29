"""Load the YAML config, falling back to config.example.yaml."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULT_FILES = ("config.yaml", "config.example.yaml")


def load_config(path: str | None = None) -> dict:
    candidates = [path] if path else list(DEFAULT_FILES)
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            with open(candidate, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            cfg["_path"] = str(candidate)
            return cfg
    raise FileNotFoundError(f"No config found (tried {', '.join(c for c in candidates if c)})")


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader so secrets stay out of the config file."""
    if not Path(path).is_file():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def instrument(cfg: dict, name: str) -> dict:
    for inst in cfg["instruments"]:
        if inst["name"].lower() == name.lower():
            return inst
    raise KeyError(f"Unknown instrument {name!r}")
