"""Small IO helpers shared by the data / eval / submit layers."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_rows(path: PathLike) -> list[dict[str, str]]:
    """Read a CSV as a list of dicts, preserving column order and raw strings."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(
    path: PathLike,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str],
    *,
    encoding: str = "utf-8",
    newline: str = "\n",
) -> Path:
    """Write CSV with LF newlines and no BOM (competition-safe defaults)."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding=encoding) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator=newline)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_json(path: PathLike, payload: object, *, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8"
    )
    return path


def read_json(path: PathLike) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "ensure_dir",
    "read_csv_rows",
    "write_csv_rows",
    "write_json",
    "read_json",
]
