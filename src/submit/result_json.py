"""Build ``result.json`` in the exact shape the official parser accepts.

The contract (from the official B-board instructions and the frozen
``validate_and_convert_submission``):

* top level: exactly ``{"version": "1.0", "results": [...]}``
* one record per test image, every image exactly once, none missing or extra
* each record: exactly ``{"image_id", "instances"}``
* ``image_id`` is the full file name **including** ``.jpg``
* each instance: exactly ``{"category_id", "score", "segmentation"}``
* ``category_id`` in the submission numbering (see ``competition.submit_classes``)
  - note this is
  *not* the training annotation numbering
* ``segmentation`` is a COCO **compressed** RLE object ``{"size", "counts"}``

An empty ``instances`` list is valid and must not be omitted: a missing record
invalidates the whole submission rather than just that image.

This module validates everything it writes, so a malformed submission fails
locally instead of on the leaderboard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from common.config import Config, load_config
from common.io_utils import write_json


class ResultJsonError(ValueError):
    """Raised when predictions cannot be expressed as a valid ``result.json``."""


@dataclass(frozen=True)
class TestImage:
    """One test image as listed in the public manifest."""

    # Not a pytest test class, despite the name.
    __test__ = False

    image_id: str      # full file name, e.g. "sample.jpg"
    width: int
    height: int


def encode_mask_to_rle(mask, *, as_fortran: bool = True) -> dict:
    """Encode a 2-D binary mask to a compressed COCO RLE dict.

    ``counts`` is decoded to ``str`` because the JSON contract requires a
    string; ``pycocotools`` returns ``bytes``.
    """
    import numpy as np
    from pycocotools import mask as mask_utils

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ResultJsonError(f"mask must be 2-D, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = (array > 0).astype(np.uint8)
    if not array.any():
        raise ResultJsonError("mask is empty; drop the instance instead of encoding it")
    rle = mask_utils.encode(np.asfortranarray(array) if as_fortran else array)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def load_test_manifest(path: str | Path) -> list[TestImage]:
    """Read a public image manifest (``image_id`` / ``width`` / ``height``)."""
    import json

    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ResultJsonError(f"{path}: expected a non-empty JSON list")
    images: list[TestImage] = []
    seen: set[str] = set()
    for row in rows:
        if set(row) != {"image_id", "width", "height"}:
            raise ResultJsonError(f"{path}: entry keys must be image_id/width/height, got {sorted(row)}")
        name = row["image_id"]
        if not isinstance(name, str) or not name or name in seen:
            raise ResultJsonError(f"{path}: bad or duplicate image_id {name!r}")
        if type(row["width"]) is not int or type(row["height"]) is not int:
            raise ResultJsonError(f"{path}: width/height must be int for {name}")
        if row["width"] <= 0 or row["height"] <= 0:
            raise ResultJsonError(f"{path}: width/height must be positive for {name}")
        seen.add(name)
        images.append(TestImage(name, row["width"], row["height"]))
    return images


def _validate_instance(instance: Mapping, cfg: Config, location: str) -> dict:
    valid = {int(value) for value in cfg.competition.submit_labels}
    if not isinstance(instance, dict):
        raise ResultJsonError(f"{location} must be an object")
    extra = set(instance) - {"category_id", "score", "segmentation"}
    missing = {"category_id", "score", "segmentation"} - set(instance)
    if extra or missing:
        raise ResultJsonError(
            f"{location} keys must be exactly category_id/score/segmentation "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )

    category_id = instance["category_id"]
    if type(category_id) is not int or category_id not in valid:
        raise ResultJsonError(
            f"{location}.category_id must be an int in {sorted(valid)}, got {category_id!r}"
        )

    score = instance["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ResultJsonError(f"{location}.score must be a number, got {type(score).__name__}")
    if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
        raise ResultJsonError(f"{location}.score must be finite in [0,1], got {score!r}")

    segmentation = instance["segmentation"]
    if not isinstance(segmentation, dict) or set(segmentation) != {"size", "counts"}:
        raise ResultJsonError(
            f"{location}.segmentation must be exactly {{size, counts}} "
            "(COCO compressed RLE), got "
            f"{sorted(segmentation) if isinstance(segmentation, dict) else type(segmentation).__name__}"
        )
    size = segmentation["size"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(value) is not int for value in size)
        or size[0] <= 0
        or size[1] <= 0
    ):
        raise ResultJsonError(f"{location}.segmentation.size must be [height, width] ints, got {size!r}")
    counts = segmentation["counts"]
    if not isinstance(counts, str) or not counts:
        raise ResultJsonError(f"{location}.segmentation.counts must be a non-empty string")

    return {
        "category_id": int(category_id),
        "score": float(score),
        "segmentation": {"size": [int(size[0]), int(size[1])], "counts": counts},
    }


def build_result(
    images: Sequence[TestImage],
    predictions: Mapping[str, Sequence[Mapping]],
    cfg: Config | None = None,
    *,
    max_instances_per_image: int | None = None,
    drop_invalid: bool = False,
) -> dict:
    """Assemble the ``result.json`` payload, validating every field.

    ``predictions`` maps ``image_id`` (full file name) to a sequence of
    instances already in submission numbering. Images with no predictions are
    emitted with ``instances: []``; an image present in ``predictions`` but not
    in ``images`` is an error (it would add an out-of-scope record).
    """
    cfg = cfg or load_config()
    limit = (
        int(max_instances_per_image)
        if max_instances_per_image is not None
        else int(cfg.inference.max_detections_per_image)
    )
    if limit <= 0:
        raise ResultJsonError(f"max_instances_per_image must be positive, got {limit}")

    known = {image.image_id for image in images}
    unknown = sorted(set(predictions) - known)
    if unknown:
        raise ResultJsonError(
            f"{len(unknown)} prediction key(s) are not test images: {unknown[:5]}"
        )

    results = []
    for image in images:
        raw_instances = list(predictions.get(image.image_id, ()))
        cleaned: list[dict] = []
        for index, instance in enumerate(raw_instances):
            try:
                cleaned.append(
                    _validate_instance(instance, cfg, f"{image.image_id}.instances[{index}]")
                )
            except ResultJsonError:
                if not drop_invalid:
                    raise
        # Highest score first, then truncate: COCO scoring takes the top-N.
        cleaned.sort(key=lambda item: item["score"], reverse=True)
        if len(cleaned) > limit:
            cleaned = cleaned[:limit]
        results.append({"image_id": image.image_id, "instances": cleaned})

    return {"version": str(cfg.submit.result_version), "results": results}


def write_result_json(payload: Mapping, path: str | Path) -> Path:
    """Write the payload as UTF-8 JSON (no BOM, LF)."""
    return write_json(path, payload)


def build_and_write(
    images: Sequence[TestImage],
    predictions: Mapping[str, Sequence[Mapping]],
    path: str | Path,
    cfg: Config | None = None,
    *,
    max_instances_per_image: int | None = None,
    drop_invalid: bool = False,
) -> tuple[Path, dict]:
    payload = build_result(
        images,
        predictions,
        cfg,
        max_instances_per_image=max_instances_per_image,
        drop_invalid=drop_invalid,
    )
    return write_result_json(payload, path), payload


def empty_result(images: Sequence[TestImage], cfg: Config | None = None) -> dict:
    """A format-valid submission with zero predictions (official score 0).

    This is the safety net: a guaranteed-valid submission exists from day one,
    so a late failure can never leave the team with nothing on the board.
    """
    return build_result(images, {}, cfg)
