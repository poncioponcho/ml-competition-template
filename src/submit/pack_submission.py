"""Pack ``b_submission.zip`` with an exactly-flat root.

The platform accepts only a zip whose root contains exactly two regular files::

    b_submission.zip
    ├── result.json
    └── solution_commit.txt

The official reader (``src/submit/official/submission.py::read_submission``)
rejects anything else outright - a directory entry, a symlink, an encrypted
member, a duplicate name, a third file, or a zero-length file all invalidate the
submission. Those conditions are therefore refused here at pack time rather than
discovered at upload time.

Member order and timestamps are normalised so that repacking identical inputs
produces identical bytes, which matters because the *solution* archive's hash is
declared separately and must not drift.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Mapping

from common.config import Config, load_config

# Fixed timestamp for reproducible archives (zip epoch: 1980-01-01).
_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)


class PackError(ValueError):
    """Raised when the archive would not satisfy the platform's reader."""


def pack_b_submission(
    result_json: str | Path,
    commit_txt: str | Path,
    out_zip: str | Path,
    cfg: Config | None = None,
) -> Path:
    """Write ``b_submission.zip`` containing exactly the two required files."""
    cfg = cfg or load_config()
    result_json = Path(result_json)
    commit_txt = Path(commit_txt)
    out = Path(out_zip)

    for path in (result_json, commit_txt):
        if not path.is_file():
            raise PackError(f"missing input file: {path}")
        if path.stat().st_size == 0:
            raise PackError(f"input file is empty: {path} (the platform rejects empty members)")

    limits = {
        str(cfg.submit.result_json_name): int(cfg.submit.max_result_bytes),
        str(cfg.submit.commit_name): int(cfg.submit.max_commit_bytes),
    }
    for path in (result_json, commit_txt):
        limit = limits.get(path.name)
        if limit is not None and path.stat().st_size > limit:
            raise PackError(
                f"{path.name} is {path.stat().st_size:,} bytes, over the {limit:,} limit"
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in (result_json, commit_txt):
            info = zipfile.ZipInfo(filename=source.name, date_time=_FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16  # regular file, not a directory
            archive.writestr(info, source.read_bytes())

    if out.stat().st_size > int(cfg.submit.max_archive_bytes):
        raise PackError(
            f"archive is {out.stat().st_size:,} bytes, over the "
            f"{int(cfg.submit.max_archive_bytes):,} limit"
        )

    expected = {str(name) for name in cfg.submit.b_root_files}
    with zipfile.ZipFile(out) as archive:
        actual = set(archive.namelist())
    if actual != expected:
        raise PackError(f"packed archive root {sorted(actual)} != {sorted(expected)}")

    return out


def pack_files(files: Mapping[str, Path], out_zip: str | Path, *, flat: bool = False) -> Path:
    """Pack ``{arcname: source_path}`` into a zip.

    ``flat=True`` (used for ``b_submission.zip``) refuses any arcname containing
    a path separator. ``flat=False`` (used for the nested ``solution.zip``
    layout ``model/ config/ src/ inference.py ...``) allows directories but
    still rejects absolute paths and traversal.
    """
    out = Path(out_zip)
    if not files:
        raise PackError("nothing to pack")
    for name, source in files.items():
        if name.startswith("/") or name.startswith("\\"):
            raise PackError(f"member name {name!r} is absolute")
        if ".." in Path(name).parts:
            raise PackError(f"member name {name!r} escapes the archive root")
        if flat and ("/" in name or "\\" in name):
            raise PackError(f"member name {name!r} contains a path separator; root must be flat")
        if not Path(source).is_file():
            raise PackError(f"missing source file for {name!r}: {source}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(filename=name, date_time=_FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, Path(files[name]).read_bytes())
    return out


def pack_solution_zip(
    files: Mapping[str, Path],
    out_zip: str | Path,
    cfg: Config | None = None,
) -> Path:
    """Pack the *solution* archive (inference code + weights + config).

    ⚠️ Hash this file **once**, ship that exact file, and never repack it: the
    declaration is bound to its bytes. See ``docs/official_rules.md``.
    """
    cfg = cfg or load_config()
    out = Path(out_zip)
    if out.name != str(cfg.submit.commit_solution_name):
        raise PackError(
            f"the solution archive must be named {cfg.submit.commit_solution_name!r}, "
            f"got {out.name!r}"
        )
    packed = pack_files(files, out)
    limit = int(cfg.submit.max_solution_zip_bytes)
    if packed.stat().st_size > limit:
        raise PackError(
            f"solution.zip is {packed.stat().st_size:,} bytes, over the {limit:,} limit"
        )
    return packed
