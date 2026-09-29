"""Guards for the multi-scale inference contract.

Two places have to agree on what ``--short_side`` means:

* ``scripts/predict_test.py`` - produces the submitted ``result.json``
* ``solution/model/model.py`` - ships inside ``solution.zip`` and must reproduce
  that same ``result.json`` offline after the competition (Mask mAP within
  0.005)

If they drift apart - e.g. the package keeps running one scale while the
submission was built from two - the reproduction silently scores differently and
the submission is disqualified. These tests pin both parsers to the same
behaviour, including the empty and whitespace cases.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def predict_module():
    return _load(REPO_ROOT / "scripts" / "predict_test.py", "predict_test_under_test")


@pytest.fixture(scope="module")
def packaged_module():
    path = REPO_ROOT / "solution" / "model" / "model.py"
    if not path.is_file():
        pytest.skip(
            "solution/model/model.py not present yet - implement the packaged "
            "model (see solution/README.md) and these contract tests activate"
        )
    return _load(path, "packaged_model_under_test")


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("1024", [1024]),
        ("640,1024", [640, 1024]),
        ("640, 1024", [640, 1024]),
        (" 640 ,1024 ", [640, 1024]),
        ([640, 1024], [640, 1024]),
    ],
)
def test_pipeline_parses_resolution_lists(predict_module, value, expected):
    assert predict_module.parse_short_sides(value) == expected


def test_packaged_and_pipeline_agree_on_resolutions(predict_module, packaged_module):
    """Same inputs must mean the same thing on both sides of the archive."""
    for value in (None, "640", "1024", "640,1024", [640, 1024]):
        pipeline = predict_module.parse_short_sides(value)
        trained_at = 640
        packaged = packaged_module._as_resolutions(value, trained_at)
        if pipeline is None:
            # The pipeline's None means "use the checkpoint's training size",
            # which is exactly what the packaged default resolves to.
            assert packaged == [trained_at]
        else:
            assert packaged == pipeline, value


def test_empty_selection_is_rejected_by_the_package(packaged_module):
    with pytest.raises(ValueError):
        packaged_module._as_resolutions("", 640)
    with pytest.raises(ValueError):
        packaged_module._as_resolutions([], 640)


def test_submitted_resolutions_are_declared(packaged_module):
    """The package must state the scales the submission was produced with."""
    declared = packaged_module.DEFAULT_INFER_SHORT_SIDES
    assert isinstance(declared, tuple) and declared
    assert all(isinstance(v, int) and v > 0 for v in declared)
    # 640 and 1024 are the two scales E10 measured; both must be present.
    assert set(declared) == {640, 1024}
