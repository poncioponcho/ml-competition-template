"""End-to-end submission pipeline, validated by the frozen official validator.

The point of this file is that "it passes" means *the platform will parse it*,
because the last assertion in each case is the official validator itself, run
as a subprocess on the exact bytes we would upload.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from submit.pack_submission import PackError, pack_b_submission, pack_solution_zip
from submit.result_json import empty_result, write_result_json
from submit.solution_commit import build_commit, write_commit
from submit.verify_submission import verify_submission

MINIMAL_SOLUTION = {
    "inference.py": b"print('inference')\n",
    "requirements.txt": b"torch==2.2.0\n",
    "README.md": b"# solution\n",
}


def _solution_files(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "solution_src"
    root.mkdir(exist_ok=True)
    files: dict[str, Path] = {}
    for name, payload in MINIMAL_SOLUTION.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files[name] = path
    return files


def _build(tmp_path: Path, cfg, *, predictions=None, flat=True) -> tuple[Path, Path, Path]:
    from submit.result_json import build_result, load_test_manifest

    manifest = cfg.path("data.test_b_manifest")
    images = load_test_manifest(manifest)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)

    solution_zip = pack_solution_zip(_solution_files(tmp_path), out / "solution.zip", cfg)
    commit = build_commit(solution_zip, cfg)
    commit_path = write_commit(commit, out / "solution_commit.txt", cfg)

    payload = build_result(images, predictions or {}, cfg)
    result_path = write_result_json(payload, out / "result.json")
    archive = pack_b_submission(result_path, commit_path, out / "b_submission.zip", cfg)
    return archive, solution_zip, result_path


def test_packed_archive_root_is_exactly_two_regular_files(tmp_path, cfg, data_available) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    archive, _, _ = _build(tmp_path, cfg)
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        assert sorted(i.filename for i in infos) == ["result.json", "solution_commit.txt"]
        assert not any(i.is_dir() for i in infos)
        assert all(i.file_size > 0 for i in infos)
        assert not any(i.flag_bits & 1 for i in infos)


def test_empty_submission_passes_the_official_validator(tmp_path, cfg, data_available) -> None:
    """The safety net: a format-valid, zero-prediction submission."""
    if not data_available:
        pytest.skip("competition data not present")
    archive, _, _ = _build(tmp_path, cfg)
    result = verify_submission(archive, cfg.path("data.test_b_manifest"), cfg)
    assert result.ok, result.report
    assert result.official_ok, result.official_stdout
    assert result.n_images == int(cfg.competition.n_test_b_images)
    assert result.n_instances == 0


def test_submission_with_predictions_passes_the_official_validator(
    tmp_path, cfg, data_available
) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    import numpy as np

    from submit.result_json import encode_mask_to_rle, load_test_manifest

    images = load_test_manifest(cfg.path("data.test_b_manifest"))
    mask = np.zeros((1152, 2048), dtype=np.uint8)
    mask[100:400, 200:600] = 1
    rle = encode_mask_to_rle(mask)
    predictions = {
        images[0].image_id: [
            {"category_id": 0, "score": 0.9, "segmentation": rle},
            {"category_id": 1, "score": 0.4, "segmentation": rle},
        ]
    }
    archive, _, _ = _build(tmp_path, cfg, predictions=predictions)
    result = verify_submission(archive, cfg.path("data.test_b_manifest"), cfg)
    assert result.ok, result.report
    assert result.official_ok, result.official_stdout
    assert result.n_instances == 2


def test_local_checks_catch_a_nested_root(tmp_path, cfg, data_available) -> None:
    """A nested archive must be rejected before upload."""
    if not data_available:
        pytest.skip("competition data not present")
    archive, _, result_path = _build(tmp_path, cfg)
    commit_path = archive.parent / "solution_commit.txt"
    nested = archive.parent / "nested.zip"
    with zipfile.ZipFile(nested, "w") as zf:
        zf.writestr("submit/result.json", result_path.read_bytes())
        zf.writestr("submit/solution_commit.txt", commit_path.read_bytes())

    result = verify_submission(nested, cfg.path("data.test_b_manifest"), cfg, run_official=False)
    assert not result.ok
    assert any("root must contain exactly" in err for err in result.errors)


def test_local_checks_catch_a_bad_commit(tmp_path, cfg, data_available) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    archive, _, result_path = _build(tmp_path, cfg)
    bad_commit = archive.parent / "solution_commit.txt"
    bad_commit.write_text("solution_name=solution.zip\n", encoding="utf-8")

    broken = archive.parent / "broken.zip"
    with zipfile.ZipFile(broken, "w") as zf:
        zf.writestr("result.json", result_path.read_bytes())
        zf.writestr("solution_commit.txt", bad_commit.read_bytes())

    result = verify_submission(broken, cfg.path("data.test_b_manifest"), cfg, run_official=False)
    assert not result.ok
    assert any("solution_commit.txt" in err for err in result.errors)


def test_local_checks_catch_a_result_missing_images(tmp_path, cfg, data_available) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    partial = {"version": "1.0", "results": [{"image_id": "sample.jpg", "instances": []}]}
    result_path = write_result_json(partial, tmp_path / "result.json")
    commit = build_commit(_pack_solution(tmp_path, cfg), cfg)
    commit_path = write_commit(commit, tmp_path / "solution_commit.txt", cfg)
    archive = pack_b_submission(result_path, commit_path, tmp_path / "b.zip", cfg)

    result = verify_submission(archive, cfg.path("data.test_b_manifest"), cfg, run_official=False)
    assert not result.ok
    assert any("missing" in err for err in result.errors)


def _pack_solution(tmp_path: Path, cfg) -> Path:
    return pack_solution_zip(_solution_files(tmp_path), tmp_path / "solution.zip", cfg)


def test_verify_rejects_an_archive_with_a_third_file(tmp_path, cfg, data_available) -> None:
    """The platform reads exactly two members; a third invalidates the upload."""
    if not data_available:
        pytest.skip("competition data not present")
    result_path = write_result_json(empty_result([], cfg), tmp_path / "result.json")
    commit_path = tmp_path / "solution_commit.txt"
    commit_path.write_text("a=1\n", encoding="utf-8")
    archive = tmp_path / "three.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("result.json", result_path.read_bytes())
        zf.writestr("solution_commit.txt", commit_path.read_bytes())
        zf.writestr("notes.txt", b"nope")

    result = verify_submission(archive, cfg.path("data.test_b_manifest"), cfg, run_official=False)
    assert not result.ok
    assert any("root must contain exactly" in err for err in result.errors)


def test_pack_rejects_empty_members(tmp_path, cfg) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    commit_path = tmp_path / "solution_commit.txt"
    commit_path.write_bytes(b"")
    with pytest.raises(PackError, match="empty"):
        pack_b_submission(result_path, commit_path, tmp_path / "b.zip", cfg)


def test_pack_is_reproducible(tmp_path, cfg) -> None:
    """Identical inputs must produce byte-identical archives."""
    result_path = write_result_json(empty_result([], cfg), tmp_path / "result.json")
    commit_path = tmp_path / "solution_commit.txt"
    commit_path.write_text("a=1\n", encoding="utf-8")
    first = pack_b_submission(result_path, commit_path, tmp_path / "one.zip", cfg)
    second = pack_b_submission(result_path, commit_path, tmp_path / "two.zip", cfg)
    assert first.read_bytes() == second.read_bytes()


def test_verify_requires_the_archive_to_exist(cfg, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_submission(tmp_path / "nope.zip", cfg.path("data.test_b_manifest"), cfg)
