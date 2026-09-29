"""SHA-256 helpers (reused from the HardLane repo's ``src/common/checksum.py``).

Used to freeze the official evaluation contract and to fingerprint every
artifact that feeds a reported number, so any silent change becomes visible.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, Path]
_CHUNK = 1 << 20


def sha256_file(path: PathLike) -> str:
    """Hex SHA-256 of a file's exact bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_files(paths: Iterable[PathLike]) -> str:
    """Order-sensitive digest over several files (name + bytes each)."""
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def assert_sha256(path: PathLike, expected: str) -> None:
    """Raise AssertionError if a frozen file changed (Oracle-freeze pattern)."""
    actual = sha256_file(path)
    if actual != expected:
        raise AssertionError(
            f"frozen artifact changed: {path}\n  expected {expected}\n  actual   {actual}"
        )


__all__ = ["sha256_file", "sha256_bytes", "sha256_text", "sha256_files", "assert_sha256"]
