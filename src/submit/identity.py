"""Competition identity guards — so the wrong archive never gets uploaded.

A workspace often holds more than one competition, and two of them may share
both the task type and the submission file name (e.g. two instance-segmentation
tasks that both expect ``result.json``). That is exactly the situation where a
correct submission gets uploaded to the wrong leaderboard - unrecoverable if it
burns one of a small number of submission attempts.

Three defences, all cheap:

1. **Pre-flight assertions** (:func:`preflight`) — before an archive is built,
   prove that the data it was produced from really is this competition's data:
   the test manifest must have the expected image count, the file names must
   match the expected pattern, and every referenced image must exist on disk.

2. **An identity banner** printed at build time, stating the competition id,
   the official URL, and the three competitions this is *not*.

3. **A sidecar record** written next to (never inside) the archive, binding the
   competition id to the archive's SHA-256. The platform forbids extra files
   inside ``b_submission.zip``, so the sidecar lives alongside it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from common.config import Config, load_config
from common.checksum import sha256_file


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str

    def render(self) -> str:
        return f"  [{'OK ' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


class IdentityError(RuntimeError):
    """Raised when the data does not belong to the configured competition."""


def _expected_images(cfg: Config, split: str) -> int:
    key = "n_test_b_images" if split.lower() == "testb" else "n_test_a_images"
    return int(cfg.competition[key])


def preflight(cfg: Config | None = None, *, split: str = "testB") -> list[PreflightCheck]:
    """Prove the local data matches the configured competition.

    Every check reads only local files: no network, no GPU.
    """
    cfg = cfg or load_config()
    from submit.result_json import load_test_manifest

    checks: list[PreflightCheck] = []

    competition_id = str(cfg.competition.get("id", ""))
    checks.append(
        PreflightCheck(
            "competition id configured",
            bool(competition_id),
            competition_id or "MISSING - set competition.id in configs/default.yaml",
        )
    )

    manifest_key = "data.test_b_manifest" if split.lower() == "testb" else "data.test_a_manifest"
    manifest_path = cfg.path(manifest_key)
    checks.append(
        PreflightCheck(
            "test manifest exists",
            manifest_path.is_file(),
            str(manifest_path),
        )
    )
    if not manifest_path.is_file():
        return checks

    images = load_test_manifest(manifest_path)
    expected = _expected_images(cfg, split)
    checks.append(
        PreflightCheck(
            f"{split} image count",
            len(images) == expected,
            f"manifest has {len(images)}, config expects {expected}",
        )
    )

    expected_size = cfg.competition.image_size
    exp_w, exp_h = int(expected_size[0]), int(expected_size[1])
    wrong_size = [i.image_id for i in images if (i.width, i.height) != (exp_w, exp_h)]
    checks.append(
        PreflightCheck(
            "image dimensions",
            not wrong_size,
            f"all {len(images)} are {exp_w}x{exp_h}"
            if not wrong_size
            else f"{len(wrong_size)} off-size, e.g. {wrong_size[:3]}",
        )
    )

    # Some competitions ship every image twice - an original plus a transformed
    # copy. That pairing is unique to the competition and is the strongest
    # identity signal available, but not every competition has it, so the check
    # is opt-in: set ``competition.n_test_b_sources`` to the number of source
    # images to enable it, and ``data.source_group.aug_suffix`` says how the
    # copy is named.
    expected_pairs = int(cfg.competition.get("n_test_b_sources", 0) or 0)
    aug_suffix = str(cfg.data.get("source_group", {}).get("aug_suffix", "") or "")
    if split.lower() == "testb" and expected_pairs and aug_suffix:
        aug = [i for i in images if Path(i.image_id).stem.endswith(aug_suffix)]
        base = [i for i in images if not Path(i.image_id).stem.endswith(aug_suffix)]
        checks.append(
            PreflightCheck(
                f"augmented pairing ({expected_pairs} + {expected_pairs})",
                len(aug) == expected_pairs and len(base) == expected_pairs,
                f"{len(base)} originals + {len(aug)} '{aug_suffix}' copies",
            )
        )

    image_dir = cfg.path("data.test_b_images" if split.lower() == "testb" else "data.test_a_images")
    if image_dir.is_dir():
        missing = [i.image_id for i in images if not (image_dir / i.image_id).is_file()]
        checks.append(
            PreflightCheck(
                "test images present on disk",
                not missing,
                f"all {len(images)} found under {image_dir.name}"
                if not missing
                else f"{len(missing)} missing, e.g. {missing[:3]}",
            )
        )
    else:
        checks.append(
            PreflightCheck("test image dir exists", False, str(image_dir))
        )

    return checks


def assert_preflight(cfg: Config | None = None, *, split: str = "testB") -> list[PreflightCheck]:
    """Run :func:`preflight` and raise unless everything passes."""
    checks = preflight(cfg, split=split)
    failed = [c for c in checks if not c.ok]
    if failed:
        detail = "\n".join(c.render() for c in failed)
        raise IdentityError(
            "pre-flight failed: this data does not look like the configured "
            f"competition.\n{detail}"
        )
    return checks


def banner(cfg: Config | None = None, *, split: str = "testB") -> str:
    """A block that must be matched against the platform upload page."""
    cfg = cfg or load_config()
    lines = [
        "",
        "=" * 74,
        "  SUBMISSION IDENTITY  —  CHECK THIS AGAINST THE UPLOAD PAGE",
        "=" * 74,
        f"  competition id : {cfg.competition.get('id', '?')}",
        f"  name           : {cfg.competition.name}",
        f"  official page  : {cfg.competition.official_page}",
        f"  split          : {split}",
        f"  archive        : {cfg.submit.b_zip_name}  "
        f"(root = {', '.join(cfg.submit.b_root_files)})",
        "-" * 74,
        "  This is NOT:",
    ]
    for entry in cfg.competition.get("not_this_competition", []) or []:
        lines.append(f"    - {entry.get('id')}: {entry.get('name')}")
        lines.append(f"      why it is different: {entry.get('why')}")
    lines.append("=" * 74)
    return "\n".join(lines)


def render_identity_record(
    cfg: Config,
    *,
    archive: Path,
    split: str,
    commit_sha256: str,
    commit_size: int,
    n_images: int,
    n_instances: int,
    checks: Sequence[PreflightCheck],
) -> str:
    """Markdown sidecar binding this archive to this competition."""
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# Submission identity record",
        "",
        "> This file lives **next to** the archive, never inside it. The platform",
        "> accepts exactly `result.json` + `solution_commit.txt` in the root and",
        "> rejects any extra file.",
        "",
        f"- generated: `{now}`",
        f"- competition id: **`{cfg.competition.get('id', '?')}`**",
        f"- competition name: {cfg.competition.name}",
        f"- official page: {cfg.competition.official_page}",
        f"- split: `{split}`",
        f"- upload this file: `{archive.name}`",
        f"- images: {n_images}   instances: {n_instances}",
        "",
        "## Declared solution package",
        "",
        f"- `solution_sha256`: `{commit_sha256}`",
        f"- `solution_size`: `{commit_size}`",
        f"- archive sha256: `{sha256_file(archive)}`",
        "",
        "## Pre-flight checks",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check.name} | {'PASS' if check.ok else 'FAIL'} | {check.detail} |"
        )
    lines += [
        "",
        "## Do not upload this to",
        "",
    ]
    for entry in cfg.competition.get("not_this_competition", []) or []:
        lines.append(f"- **{entry.get('id')}** {entry.get('name')} — {entry.get('why')}")
    lines += [
        "",
        "## Reminder",
        "",
        "- The B board allows **3 submissions total**; a mis-uploaded file burns one.",
        "- After a successful upload, confirm the platform shows the expected",
        "  competition and that the submission is recorded as valid.",
        "- Do **not** re-zip `solution.zip` afterwards: the declaration is bound to",
        "  the original bytes.",
        "",
    ]
    return "\n".join(lines)


def write_identity_record(cfg: Config, out_dir: Path, **kwargs) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "SUBMISSION_IDENTITY.md"
    path.write_text(render_identity_record(cfg, **kwargs), encoding="utf-8")
    return path


def main() -> None:  # pragma: no cover - CLI glue
    """Print the identity banner and run the pre-flight checks.

    Exits non-zero if anything fails, so it chains with ``&&`` before an upload.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["testB", "testA"], default="testB")
    args = parser.parse_args()

    cfg = load_config()
    print(banner(cfg, split=args.split))
    checks = preflight(cfg, split=args.split)
    for check in checks:
        print(check.render())
    failed = [c for c in checks if not c.ok]
    print()
    if failed:
        print(f"❌ {len(failed)} check(s) FAILED - do not upload")
        raise SystemExit(1)
    print(f"✅ all {len(checks)} checks passed - data matches {cfg.competition.get('id')}")
    raise SystemExit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
