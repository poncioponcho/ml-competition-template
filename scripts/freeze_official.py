#!/usr/bin/env python
"""Record the SHA-256 of the frozen official scripts.

Why this exists
---------------
An evaluation script that can be edited is an evaluation script that *will* be
edited, and then every number it produced becomes unverifiable. So the official
scorer/validator are copied into ``src/`` byte-for-byte, their SHA-256 is
recorded here, and ``tests/test_official_freeze.py`` turns any later edit into a
red light.

Usage
-----
    # after extracting the official kit into src/eval/official_oracle/ and
    # src/submit/official/
    python scripts/freeze_official.py
    python scripts/freeze_official.py --source data/raw/official_kit.zip

``--source`` records provenance (``<archive>::<member>``) so a future reader can
tell where each frozen file came from. Without it the provenance is left empty
and the freeze test's provenance assertion is skipped for those entries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.checksum import sha256_file  # noqa: E402

FROZEN_DIRS = ("src/eval/official_oracle", "src/submit/official")
MANIFEST = REPO_ROOT / "src" / "official_freeze.json"


def collect(frozen_dirs: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for relative in frozen_dirs:
        directory = REPO_ROOT / relative
        if not directory.is_dir():
            continue
        files.extend(
            path for path in sorted(directory.rglob("*.py"))
            if path.is_file() and "__pycache__" not in path.parts
        )
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None,
                        help="archive the official files came from, recorded as "
                             "provenance (<archive>::<member>)")
    parser.add_argument("--member-template", default="{name}",
                        help="member name pattern inside --source (default: {name})")
    args = parser.parse_args()

    files = collect(FROZEN_DIRS)
    if not files:
        raise SystemExit(
            "no official scripts found under " + " / ".join(FROZEN_DIRS) + "\n"
            "extract them from the official submission kit first (see TEMPLATE.md)"
        )

    frozen: dict[str, dict] = {}
    for path in files:
        relative = path.relative_to(REPO_ROOT).as_posix()
        record = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "source": (
                f"{args.source}::{args.member_template.format(name=path.name)}"
                if args.source else ""
            ),
        }
        frozen[relative] = record
        print(f"  {relative}\n    sha256={record['sha256'][:16]}... bytes={record['bytes']}")

    MANIFEST.write_text(
        json.dumps({"frozen": frozen}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {MANIFEST.relative_to(REPO_ROOT)} ({len(frozen)} files)")
    print("next: make test   # the freeze test must pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
