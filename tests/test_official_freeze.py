"""The frozen official scripts must not change.

Rationale: an evaluation script that can be edited is an evaluation script that
will be edited, and then every number it produced becomes unverifiable.
Recording the SHA-256 at freeze time and asserting it in a test turns any
accidental edit into a red light.

Regenerate the manifest after extracting the official kit:

    python scripts/freeze_official.py --source <official_kit.zip>

The frozen copies live outside version control when the repository is public
(they are the organiser's own code). A checkout without them - or a project
that has not frozen anything yet - skips these tests rather than failing, so a
fresh clone is green either way.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.checksum import sha256_file

FREEZE_MANIFEST = Path(__file__).resolve().parents[1] / "src" / "official_freeze.json"

NOT_FROZEN_HINT = (
    "no official scripts frozen yet - extract the official scorer/validator "
    "into src/eval/official_oracle/ and src/submit/official/, then run "
    "scripts/freeze_official.py (see TEMPLATE.md section 3)"
)
MISSING_KIT_HINT = (
    "official script not present - copy src/eval/official_oracle/ and "
    "src/submit/official/ back from the official submission kit; "
    "src/official_freeze.json records the SHA-256 each file must have"
)


@pytest.fixture(scope="module")
def frozen() -> dict:
    payload = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    return payload["frozen"]


def test_freeze_manifest_exists(frozen) -> None:
    if not frozen:
        pytest.skip(NOT_FROZEN_HINT)


def test_frozen_file_hash_matches(frozen) -> None:
    if not frozen:
        pytest.skip(NOT_FROZEN_HINT)
    repo_root = FREEZE_MANIFEST.parents[1]
    checked = 0
    for relative, record in frozen.items():
        path = repo_root / relative
        if not path.is_file():
            continue
        assert sha256_file(path) == record["sha256"], (
            f"{relative} changed since freeze time - the official scorer/validator "
            "is immutable; if the organisers published a new version, re-freeze it "
            "deliberately and log the change in docs/experiments.md"
        )
        checked += 1
    if not checked:
        pytest.skip(MISSING_KIT_HINT)


def test_frozen_file_sizes_match(frozen) -> None:
    if not frozen:
        pytest.skip(NOT_FROZEN_HINT)
    repo_root = FREEZE_MANIFEST.parents[1]
    checked = 0
    for relative, record in frozen.items():
        path = repo_root / relative
        if not path.is_file():
            continue
        assert path.stat().st_size == record["bytes"], f"{relative} size changed"
        checked += 1
    if not checked:
        pytest.skip(MISSING_KIT_HINT)


def test_frozen_entries_have_provenance(frozen) -> None:
    """Each record should point at a real source archive/member, not a rewrite."""
    if not frozen:
        pytest.skip(NOT_FROZEN_HINT)
    for relative, record in frozen.items():
        assert len(record["sha256"]) == 64, f"{relative} has a malformed digest"
        assert record["bytes"] > 0, f"{relative} has no byte count"
        # Provenance is optional (freeze_official.py --source) but if present it
        # must name the archive it came from.
        assert record["source"] == "" or "::" in record["source"], (
            f"{relative} has no provenance"
        )
