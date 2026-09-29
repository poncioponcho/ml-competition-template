"""Regression guard for multi-scale fusion in ``predict_image``.

The bug this file exists for: when multi-scale support was added to
``predict_image``, the four ``*_all.append(...)`` calls that collect detections
stayed one indentation level too far out. With a single resolution that is
invisible - the loop body runs once, so the appends happen once per model and
everything lines up. With two resolutions it silently collected the boxes from
*both* scales but the scores/labels/masks from only the last one, and raised
``UnboundLocalError`` whenever a model returned no detections at all.

A test with a real checkpoint would not catch this cheaply, so this uses a stub
model: what matters is that every (model, resolution) pair contributes its
detections to the union.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
Image = pytest.importorskip("PIL.Image")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.config import load_config  # noqa: E402


def _load_predict_module():
    spec = importlib.util.spec_from_file_location(
        "predict_test_under_test", REPO_ROOT / "scripts" / "predict_test.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubModel:
    """Returns one fixed detection per call, tagged by the input resolution.

    The returned box differs per resolution, so the fusion can be checked by
    counting how many distinct scales made it into the output.
    """

    def __init__(self, score: float = 0.9):
        self.score = score
        self.calls: list[tuple[int, int]] = []

    def __call__(self, batch):
        tensor = batch[0]
        _, height, width = tensor.shape
        self.calls.append((height, width))
        # One 10x10 box in the top-left of whatever grid it was given. The mask
        # must be non-empty: predict_image drops instances whose mask is empty
        # (they cannot be encoded as RLE).
        return [{
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([self.score]),
            "labels": torch.tensor([1]),
            "masks": torch.ones((1, 1, height, width)),
        }]


@pytest.fixture
def synthetic_image(tmp_path: Path) -> Path:
    path = tmp_path / "IMG_test.jpg"
    Image.fromarray(np.zeros((64, 128, 3), dtype=np.uint8)).save(path)
    return path


def test_every_resolution_contributes_detections(synthetic_image, tmp_path):
    predict = _load_predict_module()
    cfg = load_config()
    model = _StubModel()
    resolutions = [64, 128]

    instances = predict.predict_image(
        [("stub.pt", model, resolutions)], synthetic_image, cfg, "cpu"
    )

    # One detection per (model, resolution). The stub's box is in the *resized*
    # grid, so the two scales map back to different boxes on the original image
    # ([0,0,10,10] vs [0,0,5,5], IoU 0.25) and NMS at 0.5 keeps both - which is
    # precisely the evidence that both scales reached the fusion. A silently
    # mis-indented append would have dropped one of them (or raised on the
    # resulting length mismatch).
    assert len(model.calls) == len(resolutions)
    assert len(instances) == len(resolutions)


def test_single_resolution_still_works(synthetic_image):
    predict = _load_predict_module()
    cfg = load_config()
    model = _StubModel()
    instances = predict.predict_image([("stub.pt", model, [64])], synthetic_image,
                                      cfg, "cpu")
    assert len(model.calls) == 1
    assert len(instances) == 1


def test_model_with_no_detections_at_one_scale_is_tolerated(synthetic_image):
    """The scale that returns nothing must not break the other scale's work."""
    predict = _load_predict_module()
    cfg = load_config()

    class _SilentThenLoud:
        def __init__(self):
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            tensor = batch[0]
            _, height, width = tensor.shape
            if self.calls == 1:  # first scale: nothing at all
                return [{
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.int64),
                    "masks": torch.zeros((0, 1, height, width)),
                }]
            return [{
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
                "masks": torch.ones((1, 1, height, width)),
            }]

    model = _SilentThenLoud()
    instances = predict.predict_image([("stub.pt", model, [64, 128])],
                                      synthetic_image, cfg, "cpu")
    assert model.calls == 2
    assert len(instances) == 1, "the second scale's detection should survive"
