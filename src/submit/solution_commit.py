"""Generate and validate ``solution_commit.txt``.

The B-board rules make this file mandatory on **every** submission: it declares
the SHA-256 and exact byte count of the original ``solution.zip``. A missing,
malformed, or mismatched declaration invalidates that submission, and after the
deadline the declared hash is checked against the real archive before the score
is confirmed.

Rules (from the official instructions and ``submission.py::parse_solution_commit``):

* exactly five fields, each appearing exactly once, in this order:
  ``solution_name``, ``hash_algorithm``, ``solution_sha256``,
  ``solution_size``, ``b_data_version``
* ``key=value`` per line, UTF-8, no comments, no blank lines, no padding
* ``solution_name`` must be ``solution.zip``; ``hash_algorithm`` must be ``SHA-256``
* ``solution_sha256`` is 64 hex chars; ``solution_size`` a positive integer
  with no leading zeros
* ``b_data_version`` comes from ``competition.data_version`` in the config
* at most 65,536 bytes

The hash is computed over the **raw bytes of the archive as shipped**. Re-zipping
changes timestamps and member order, which changes the digest - so the original
file must be kept, and this module deliberately refuses to hash anything else.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from common.config import Config, load_config

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SIZE_RE = re.compile(r"^[1-9][0-9]*$")


class CommitError(ValueError):
    """Raised when a declaration would be rejected by the platform."""


@dataclass(frozen=True)
class SolutionCommit:
    solution_name: str
    hash_algorithm: str
    solution_sha256: str
    solution_size: int
    b_data_version: str

    def to_text(self, cfg: Config | None = None) -> str:
        """Render in the fixed field order, LF-terminated, no trailing blank line."""
        cfg = cfg or load_config()
        order = [str(field) for field in cfg.submit.commit_fields]
        values = {
            "solution_name": self.solution_name,
            "hash_algorithm": self.hash_algorithm,
            "solution_sha256": self.solution_sha256,
            "solution_size": str(self.solution_size),
            "b_data_version": self.b_data_version,
        }
        missing = set(order) - set(values)
        if missing:
            raise CommitError(f"commit_fields in config has unknown entries: {sorted(missing)}")
        return "".join(f"{field}={values[field]}\n" for field in order)

    def to_dict(self) -> dict:
        return {
            "solution_name": self.solution_name,
            "hash_algorithm": self.hash_algorithm,
            "solution_sha256": self.solution_sha256,
            "solution_size": self.solution_size,
            "b_data_version": self.b_data_version,
        }


def hash_zip(path: str | Path) -> tuple[str, int]:
    """Stream a file and return ``(sha256_hex, byte_count)``."""
    path = Path(path)
    if not path.is_file():
        raise CommitError(f"solution archive not found: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def build_commit(solution_zip: str | Path, cfg: Config | None = None) -> SolutionCommit:
    """Compute the real declaration for a ``solution.zip``."""
    cfg = cfg or load_config()
    path = Path(solution_zip)
    if path.name != str(cfg.submit.commit_solution_name):
        raise CommitError(
            f"the archive must be named {cfg.submit.commit_solution_name!r}, "
            f"got {path.name!r} (the platform hashes it by that name)"
        )
    import zipfile

    if not zipfile.is_zipfile(path):
        raise CommitError(f"not a valid zip archive: {path}")
    sha256, size = hash_zip(path)
    return SolutionCommit(
        solution_name=str(cfg.submit.commit_solution_name),
        hash_algorithm=str(cfg.submit.commit_hash_algorithm),
        solution_sha256=sha256,
        solution_size=size,
        b_data_version=str(cfg.competition.b_data_version),
    )


def parse_commit_text(text: str, cfg: Config | None = None) -> dict:
    """Local re-implementation of the official parser, for fast feedback."""
    cfg = cfg or load_config()
    order = [str(field) for field in cfg.submit.commit_fields]
    if len(text.encode("utf-8")) > int(cfg.submit.max_commit_bytes):
        raise CommitError("solution_commit.txt exceeds the 65,536 byte limit")

    fields: dict[str, str] = {}
    lines = text.splitlines()
    if not lines:
        raise CommitError("solution_commit.txt is empty")
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise CommitError(f"line {number}: blank lines are not allowed")
        if "=" not in line:
            raise CommitError(f"line {number}: expected key=value, got {line!r}")
        key, value = line.split("=", 1)
        if key != key.strip() or value != value.strip():
            raise CommitError(f"line {number}: no padding whitespace allowed around key/value")
        if key not in order:
            raise CommitError(f"line {number}: unknown field {key!r}")
        if key in fields:
            raise CommitError(f"line {number}: duplicate field {key!r}")
        if not value:
            raise CommitError(f"line {number}: empty value for {key!r}")
        fields[key] = value

    missing = set(order) - set(fields)
    if missing:
        raise CommitError(f"missing field(s): {sorted(missing)}")
    if len(lines) != len(order):
        raise CommitError(f"expected exactly {len(order)} lines, got {len(lines)}")
    if fields["solution_name"] != str(cfg.submit.commit_solution_name):
        raise CommitError("solution_name must be solution.zip")
    if fields["hash_algorithm"] != str(cfg.submit.commit_hash_algorithm):
        raise CommitError("hash_algorithm must be SHA-256")
    if not _SHA256_RE.match(fields["solution_sha256"]):
        raise CommitError("solution_sha256 must be 64 hex characters")
    if not _SIZE_RE.match(fields["solution_size"]):
        raise CommitError("solution_size must be a positive integer with no leading zeros")
    if fields["b_data_version"] != str(cfg.competition.b_data_version):
        raise CommitError(f"b_data_version must be {cfg.competition.b_data_version}")
    return fields


def write_commit(commit: SolutionCommit, path: str | Path, cfg: Config | None = None) -> Path:
    """Validate then write, so an invalid declaration can never reach disk."""
    cfg = cfg or load_config()
    text = commit.to_text(cfg)
    parse_commit_text(text, cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def build_and_write(solution_zip: str | Path, path: str | Path, cfg: Config | None = None):
    cfg = cfg or load_config()
    commit = build_commit(solution_zip, cfg)
    return write_commit(commit, path, cfg), commit


def verify_against_solution(
    commit: SolutionCommit, solution_zip: str | Path
) -> None:
    """Post-hoc check: does the declaration still match the archive?"""
    sha256, size = hash_zip(solution_zip)
    if sha256 != commit.solution_sha256.lower():
        raise CommitError(
            "declared SHA-256 does not match the archive:\n"
            f"  declared {commit.solution_sha256}\n  actual   {sha256}"
        )
    if size != commit.solution_size:
        raise CommitError(
            f"declared size {commit.solution_size} != actual {size} bytes"
        )
