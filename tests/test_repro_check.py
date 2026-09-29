"""Pins the semantics of the reproduction check's comparison.

``repro_check.py`` decides whether a packed ``solution.zip`` may be uploaded, so
its tolerances are load-bearing: too strict and a reproducible package is
rejected (which is exactly what happened - the submitted predictions come from
MPS, the check runs on CPU), too loose and a genuinely broken package slips
through.

The rules under test:
* category ids - strict
* masks - by IoU, so a few boundary pixels may differ but not the shape
* scores - within a small tolerance (they shift in the 1e-6 range across devices)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
mask_utils = pytest.importorskip("pycocotools.mask")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(
        "repro_check_under_test", REPO_ROOT / "scripts" / "repro_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def repro():
    return _load_repro_module()


def _rle(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(encoded["size"][0]), int(encoded["size"][1])],
            "counts": counts}


def _square(size: int = 40, top: int = 5, left: int = 5, side: int = 20) -> dict:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[top:top + side, left:left + side] = 1
    return _rle(mask)


def _instance(segmentation: dict, score: float = 0.9, category_id: int = 0) -> dict:
    return {"category_id": category_id, "score": score, "segmentation": segmentation}


def test_identical_instances_pass(repro):
    item = _instance(_square())
    ok, reason, delta, iou = repro.compare_instances([item], [item])
    assert ok, reason
    assert delta == 0.0
    assert iou == 1.0


def test_score_noise_within_tolerance_passes(repro):
    """1e-6 is what MPS-vs-CPU rounding produces; it must not fail the check."""
    ok, reason, delta, _ = repro.compare_instances(
        [_instance(_square(), score=0.995232)],
        [_instance(_square(), score=0.995233)],
    )
    assert ok, reason
    assert delta == pytest.approx(1e-6)


def test_score_difference_beyond_tolerance_fails(repro):
    ok, reason, _, _ = repro.compare_instances(
        [_instance(_square(), score=0.9)], [_instance(_square(), score=0.8)]
    )
    assert not ok
    assert "score" in reason


def test_one_pixel_mask_difference_passes(repro):
    """A boundary pixel flipping is device noise, not a different prediction."""
    a = _square()
    shifted = np.zeros((40, 40), dtype=np.uint8)
    shifted[5:25, 5:25] = 1
    shifted[5, 5] = 0  # one pixel off: IoU = 399/400 = 0.9975
    ok, reason, _, iou = repro.compare_instances([_instance(a)], [_instance(_rle(shifted))])
    assert ok, reason
    assert 0.99 < iou < 1.0


def test_different_mask_shape_fails(repro):
    ok, reason, _, _ = repro.compare_instances(
        [_instance(_square(side=20))], [_instance(_square(side=10))]
    )
    assert not ok
    assert "IoU" in reason


def test_category_mismatch_fails(repro):
    ok, reason, _, _ = repro.compare_instances(
        [_instance(_square(), category_id=0)], [_instance(_square(), category_id=1)]
    )
    assert not ok
    assert "category_id" in reason


def test_instance_count_mismatch_fails(repro):
    ok, reason, _, _ = repro.compare_instances(
        [_instance(_square())], [_instance(_square()), _instance(_square(top=25))]
    )
    assert not ok
    assert "count" in reason


def test_tolerances_are_declared_and_sane(repro):
    assert 0 < repro.SCORE_TOLERANCE <= 1e-4
    assert 0.99 <= repro.MASK_IOU_TOLERANCE < 1.0
