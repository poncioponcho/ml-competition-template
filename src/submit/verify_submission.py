"""Verify a ``b_submission.zip`` before uploading it.

Two layers, deliberately:

1. **The official validator** (``validate_b_submission.py``, frozen, run as a
   subprocess) - this is the authority. If it passes, the platform will parse
   the archive.
2. **Local structural checks** - they reproduce the official reader's rules and
   add field-level diagnostics, so a failure names the exact offending entry
   instead of a single opaque message. They also run with no network and no GPU.

Neither layer can score a submission: test labels are withheld, and the official
validator says so explicitly. Passing here means "format valid", not "good".
"""
from __future__ import annotations

import json
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

from common.config import Config, load_config
from submit.result_json import load_test_manifest
from submit.solution_commit import CommitError, parse_commit_text

PathLike = Union[str, Path]
MAX_REPORTED_ERRORS = 20


@dataclass
class VerifyResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    archive: str = ""
    n_images: int = 0
    n_instances: int = 0
    official_ok: bool = False
    official_stdout: str = ""

    @property
    def report(self) -> str:
        head = f"verify {'PASS' if self.ok else 'FAIL'}  {Path(self.archive).name}"
        lines = [
            head,
            f"  images: {self.n_images}   instances: {self.n_instances}",
            f"  official validator: {'PASS' if self.official_ok else 'FAIL'}",
            f"  local checks: {len(self.checks_run)}",
        ]
        if self.errors:
            lines.append(f"  errors: {len(self.errors)}")
            lines.extend(f"    - {err}" for err in self.errors[:MAX_REPORTED_ERRORS])
            if len(self.errors) > MAX_REPORTED_ERRORS:
                lines.append(f"    ... and {len(self.errors) - MAX_REPORTED_ERRORS} more")
        else:
            lines.append("  all checks passed")
        return "\n".join(lines)

    def write_report(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.report + "\n", encoding="utf-8")
        return path


def _load_json_strict(path: Path) -> object:
    """JSON with duplicate-key rejection, matching the official loader."""

    def reject(pairs):
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise ValueError(f"duplicate JSON key {key!r}")
            seen.add(key)
        return dict(pairs)

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def local_checks(
    archive: Path, expected_image_ids: Sequence[str], cfg: Config
) -> tuple[list[str], list[str], int, int, bytes | None]:
    """Reproduce the official reader's rules with field-level diagnostics."""
    errors: list[str] = []
    checks: list[str] = []
    n_images = n_instances = 0
    result_bytes: bytes | None = None

    checks.append("zip_readable")
    if not zipfile.is_zipfile(archive):
        return ["not a readable zip archive"], checks, 0, 0, None

    wanted = {str(name) for name in cfg.submit.b_root_files}
    limits = {
        str(cfg.submit.result_json_name): int(cfg.submit.max_result_bytes),
        str(cfg.submit.commit_name): int(cfg.submit.max_commit_bytes),
    }

    checks.extend(["flat_root", "member_types", "member_sizes"])
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        names = [info.filename for info in infos]

        if archive.stat().st_size > int(cfg.submit.max_archive_bytes):
            errors.append(
                f"archive is {archive.stat().st_size:,} bytes, over the "
                f"{int(cfg.submit.max_archive_bytes):,} limit"
            )
        if len(infos) != len(wanted) or set(names) != wanted:
            errors.append(
                f"root must contain exactly {sorted(wanted)}, found {sorted(names)}"
            )
        for info in infos:
            if "/" in info.filename or "\\" in info.filename:
                errors.append(f"member {info.filename!r} is nested; the root must be flat")
            if info.is_dir():
                errors.append(f"member {info.filename!r} is a directory entry")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                errors.append(f"member {info.filename!r} is not a regular file")
            if info.flag_bits & 1:
                errors.append(f"member {info.filename!r} is encrypted")
            limit = limits.get(info.filename)
            if limit is not None and (info.file_size == 0 or info.file_size > limit):
                errors.append(
                    f"{info.filename} is empty or over its {limit:,} byte limit"
                )
            if info.orig_filename != info.filename:
                errors.append(f"member {info.filename!r} has a non-portable name")

        if str(cfg.submit.commit_name) in names:
            checks.append("solution_commit")
            raw = zf.read(str(cfg.submit.commit_name))
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                errors.append(f"{cfg.submit.commit_name} is not UTF-8: {exc}")
                text = ""
            if text:
                try:
                    parse_commit_text(text, cfg)
                except CommitError as exc:
                    errors.append(f"{cfg.submit.commit_name}: {exc}")

        if str(cfg.submit.result_json_name) in names:
            checks.append("result_json")
            result_bytes = zf.read(str(cfg.submit.result_json_name))
            try:
                payload = _load_json_strict_path(result_bytes)
            except Exception as exc:
                errors.append(f"{cfg.submit.result_json_name}: invalid JSON ({exc})")
                payload = None
            if payload is not None:
                errors.extend(_check_result_payload(payload, expected_image_ids, cfg))
                if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                    n_images = len(payload["results"])
                    n_instances = sum(
                        len(record.get("instances") or [])
                        for record in payload["results"]
                        if isinstance(record, dict)
                    )

    return errors, checks, n_images, n_instances, result_bytes


def _load_json_strict_path(data: bytes) -> object:
    def reject(pairs):
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise ValueError(f"duplicate JSON key {key!r}")
            seen.add(key)
        return dict(pairs)

    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=reject)


def _check_result_payload(payload: object, expected_ids: Sequence[str], cfg: Config) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["result.json top level must be an object"]
    if set(payload) != {"version", "results"}:
        errors.append(f"result.json keys must be exactly version/results, got {sorted(payload)}")
        return errors
    if payload["version"] != str(cfg.submit.result_version):
        errors.append(f"version must be {cfg.submit.result_version!r}, got {payload['version']!r}")

    results = payload["results"]
    if not isinstance(results, list):
        return errors + ["results must be a list"]

    expected = set(expected_ids)
    seen: set[str] = set()
    valid_categories = {int(value) for value in cfg.competition.submit_labels}
    bad_instances: list[str] = []
    for index, record in enumerate(results):
        location = f"results[{index}]"
        if not isinstance(record, dict) or set(record) != {"image_id", "instances"}:
            errors.append(f"{location} keys must be exactly image_id/instances")
            continue
        image_id = record["image_id"]
        if not isinstance(image_id, str) or not image_id:
            errors.append(f"{location}.image_id must be a non-empty string")
            continue
        if image_id not in expected:
            errors.append(f"{location}.image_id {image_id!r} is not a test image")
            continue
        if image_id in seen:
            errors.append(f"duplicate image_id {image_id!r}")
            continue
        seen.add(image_id)

        instances = record["instances"]
        if not isinstance(instances, list):
            errors.append(f"{location}.instances must be a list")
            continue
        for instance_index, instance in enumerate(instances):
            where = f"{location}.instances[{instance_index}]"
            if not isinstance(instance, dict) or set(instance) != {
                "category_id", "score", "segmentation"
            }:
                bad_instances.append(f"{where}: keys must be exactly category_id/score/segmentation")
                continue
            category = instance["category_id"]
            if type(category) is not int or category not in valid_categories:
                bad_instances.append(f"{where}.category_id must be int in {sorted(valid_categories)}")
            score = instance["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                bad_instances.append(f"{where}.score must be numeric")
            segmentation = instance["segmentation"]
            if not isinstance(segmentation, dict) or set(segmentation) != {"size", "counts"}:
                bad_instances.append(f"{where}.segmentation must be {{size, counts}}")

    missing = sorted(expected - seen)
    if missing:
        errors.append(f"{len(missing)} test image(s) missing: {missing[:5]}")
    if len(seen) != len(expected):
        errors.append(f"expected {len(expected)} image records, found {len(seen)}")
    if bad_instances:
        errors.append(f"{len(bad_instances)} malformed instance(s): {bad_instances[:3]}")
    return errors


def run_official_validator(
    archive: Path,
    manifest: PathLike,
    cfg: Config,
    *,
    python_executable: str | None = None,
    timeout: int = 900,
) -> tuple[bool, str]:
    """Run the frozen official B-board validator as a subprocess.

    The validator is an unmodified copy of the official file, so its sibling
    imports (``official_evaluate``, ``rle_validation``, ``submission``) must be
    resolvable. They live in two frozen directories; both go on ``PYTHONPATH``
    rather than being copied next to each other, so each file keeps its
    recorded SHA-256.
    """
    import os

    validator = cfg.path("submit.official_validator")
    if not Path(validator).is_file():
        return False, f"official validator not found: {validator}"

    submit_official = Path(validator).resolve().parent
    oracle_dir = cfg.path("eval.oracle_script").resolve().parent
    search_path = os.pathsep.join([str(submit_official), str(oracle_dir)])
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        search_path + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else search_path
    )

    command = [
        python_executable or sys.executable,
        str(Path(validator).resolve()),
        str(archive.resolve()),
        str(Path(manifest).resolve()),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout,
        cwd=str(submit_official), env=env,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, output.strip()


def verify_submission(
    archive: PathLike,
    manifest: PathLike,
    cfg: Config | None = None,
    *,
    report_path: PathLike | None = None,
    run_official: bool = True,
) -> VerifyResult:
    """Full verification of a candidate ``b_submission.zip``."""
    cfg = cfg or load_config()
    archive = Path(archive)
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")

    images = load_test_manifest(manifest)
    expected_ids = [image.image_id for image in images]

    errors, checks, n_images, n_instances, _ = local_checks(archive, expected_ids, cfg)

    official_ok = False
    official_output = ""
    if run_official:
        official_ok, official_output = run_official_validator(archive, manifest, cfg)
        if not official_ok:
            errors.append(f"official validator failed: {official_output.splitlines()[-1][:400] if official_output else 'no output'}")

    result = VerifyResult(
        ok=not errors and (official_ok or not run_official),
        errors=errors,
        checks_run=checks,
        archive=str(archive),
        n_images=n_images,
        n_instances=n_instances,
        official_ok=official_ok,
        official_stdout=official_output,
    )
    if report_path is not None:
        result.write_report(report_path)
    return result


def main() -> None:  # pragma: no cover - CLI glue
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--no-official", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    manifest = args.manifest or cfg.path("data.test_b_manifest")
    result = verify_submission(
        args.archive, manifest, cfg,
        report_path=args.report, run_official=not args.no_official,
    )
    print(result.report)
    if result.official_stdout:
        print("--- official validator output ---")
        print(result.official_stdout)
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
