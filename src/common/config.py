"""Config loading + project path resolution.

`configs/default.yaml` is the single source of truth for every tunable
constant (paths, thresholds, class definitions, file names). Nothing in
`src/`, `scripts/` or `tests/` may hardcode those values.

Usage
-----
>>> from common.config import load_config
>>> cfg = load_config()               # finds configs/default.yaml from repo root
>>> cfg.competition.image_size
224
>>> cfg.path("data.processed_dir")    # dotted lookup -> absolute Path
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import yaml

# Repo root = parent of src/ (this file is src/common/config.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_REL = "configs/default.yaml"


def _as_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_as_dict(item) for item in value]
    return value


class Config(dict):
    """dict with attribute access; nested dicts are wrapped recursively."""

    def __init__(self, mapping: dict):
        super().__init__({k: _as_dict(v) for k, v in mapping.items()})

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __iter__(self) -> Iterator[str]:
        return super().__iter__()

    def get_path(self, dotted: str) -> Any:
        """Resolve a dotted key path, e.g. ``cfg.get_path("competition.image_size")``."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"config key not found: {dotted!r} (failed at {part!r})")
            node = node[part]
        return node

    def path(self, dotted: str, *, root: Path | None = None) -> Path:
        """Resolve a dotted key holding a repo-relative path to an absolute Path."""
        raw = self.get_path(dotted)
        candidate = Path(str(raw))
        if candidate.is_absolute():
            return candidate
        return (root or REPO_ROOT) / candidate


@lru_cache(maxsize=8)
def _load_cached(config_path: str) -> Config:
    text = Path(config_path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping at top level")
    return Config(data)


def load_config(config_path: str | os.PathLike | None = None) -> Config:
    """Load the project config.

    Resolution order: explicit argument -> ``$BERRY_CONFIG`` -> repo default.
    """
    if config_path is None:
        config_path = os.environ.get("BERRY_CONFIG", str(REPO_ROOT / DEFAULT_CONFIG_REL))
    return _load_cached(str(Path(config_path).resolve()))


def reload_config(config_path: str | os.PathLike | None = None) -> Config:
    """Bypass the cache (used by tests that mutate the config file)."""
    _load_cached.cache_clear()
    return load_config(config_path)


__all__ = ["Config", "load_config", "reload_config", "REPO_ROOT", "DEFAULT_CONFIG_REL"]
