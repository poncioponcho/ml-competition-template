"""COCO data layer for an instance-segmentation task.

Two facts drive the design:

1. **Category renumbering is mandatory.** The training COCO annotations use
   ``category_id`` 1 and 2, but ``result.json`` must use 0 and 1. Getting this
   wrong silently maps one class onto the other and tanks the score, so the
   mapping lives in config and is applied in exactly one place.

2. **``IMG_x.jpg`` and ``IMG_x_aug1.jpg`` are the same source image.** Every
   split in this competition (train, A, B) ships both. A local validation split
   that separates the pair leaks near-duplicate content across train/val, so
   the group key strips the augmentation suffix.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from common.config import Config, load_config
from common.io_utils import read_csv_rows, write_csv_rows

MANIFEST_FIELDS = ("image_id", "file_name", "source_id", "width", "height", "n_instances", "fold")


class CocoError(ValueError):
    """Raised when the COCO data violates the competition contract."""


@dataclass(frozen=True)
class ImageRecord:
    """One image record; ``source_id`` groups augmented copies together."""

    image_id: int
    file_name: str
    width: int
    height: int
    source_id: str
    n_instances: int = 0
    fold: int = -1

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def derive_source_id(file_name: str, cfg: Config) -> str:
    """Strip the augmentation suffix and extension to get the source key.

    ``sample_aug1.jpg`` -> ``sample``; ``sample.jpg`` -> ``sample``.
    """
    group = cfg.data.source_group
    name = Path(str(file_name).replace("\\", "/")).name
    for suffix in group.strip_suffixes:
        if name.lower().endswith(str(suffix).lower()):
            name = name[: -len(str(suffix))]
            break
    aug = str(group.aug_suffix)
    if aug and name.endswith(aug):
        name = name[: -len(aug)]
    return name


def load_coco(path: str | Path) -> dict:
    """Load a COCO JSON with duplicate-key rejection."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"COCO json not found: {path}")

    def reject_duplicates(pairs):
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise CocoError(f"{path}: duplicate JSON key {key!r}")
            seen.add(key)
        return dict(pairs)

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def gt_category_to_submit(gt_category_id: int, cfg: Config) -> int:
    """Map a training COCO ``category_id`` onto the submission numbering."""
    mapping = {int(k): int(v) for k, v in cfg.competition.gt_to_submit_category.items()}
    if gt_category_id not in mapping:
        raise CocoError(
            f"unexpected GT category_id {gt_category_id}; "
            f"expected one of {sorted(mapping)}"
        )
    return mapping[gt_category_id]


def load_train_images(cfg: Config | None = None) -> list[ImageRecord]:
    """Training images with source grouping and per-image instance counts."""
    cfg = cfg or load_config()
    data = load_coco(cfg.path("data.instances_json"))

    categories = {int(c["id"]) for c in data.get("categories", [])}
    expected = {int(k) for k in cfg.competition.gt_to_submit_category}
    if categories != expected:
        raise CocoError(
            f"training categories {sorted(categories)} != expected {sorted(expected)}"
        )

    counts: dict[int, int] = {}
    for annotation in data.get("annotations", []):
        image_id = int(annotation["image_id"])
        counts[image_id] = counts.get(image_id, 0) + 1

    images: list[ImageRecord] = []
    seen: set[int] = set()
    for entry in data.get("images", []):
        image_id = int(entry["id"])
        if image_id in seen:
            raise CocoError(f"duplicate image id {image_id}")
        seen.add(image_id)
        file_name = str(entry["file_name"])
        images.append(
            ImageRecord(
                image_id=image_id,
                file_name=file_name,
                width=int(entry["width"]),
                height=int(entry["height"]),
                source_id=derive_source_id(file_name, cfg),
                n_instances=counts.get(image_id, 0),
            )
        )
    if not images:
        raise CocoError("no images found in the training annotations")
    return images


def load_train_annotations(cfg: Config | None = None) -> list[dict]:
    """Training annotations with ``category_id`` already mapped to submit ids."""
    cfg = cfg or load_config()
    data = load_coco(cfg.path("data.instances_json"))
    out: list[dict] = []
    for annotation in data.get("annotations", []):
        out.append(
            {
                "id": int(annotation["id"]),
                "image_id": int(annotation["image_id"]),
                "category_id": gt_category_to_submit(int(annotation["category_id"]), cfg),
                "segmentation": annotation["segmentation"],
                "area": float(annotation.get("area", 0.0)),
                "bbox": [float(v) for v in annotation.get("bbox", [])],
                "iscrowd": int(annotation.get("iscrowd", 0)),
            }
        )
    return out


def load_ignore_regions(cfg: Config | None = None) -> list[dict]:
    """Category-agnostic ignore regions.

    Note the non-standard key: they live under ``ignore_regions`` (not
    ``annotations``), carry no ``category_id``, and have their own ``images``
    list. They are *not* a third class.
    """
    cfg = cfg or load_config()
    path = cfg.path("data.ignore_json")
    if not Path(path).is_file():
        return []
    data = load_coco(path)
    out: list[dict] = []
    for region in data.get("ignore_regions", []):
        out.append(
            {
                "id": int(region["id"]),
                "image_id": int(region["image_id"]),
                "file_name": str(region.get("file_name", "")),
                "segmentation": region["segmentation"],
                "area": float(region.get("area", 0.0)),
                "bbox": [float(v) for v in region.get("bbox", [])],
            }
        )
    return out


def dataset_stats(cfg: Config | None = None) -> dict:
    """Diagnostics printed before training: shape of the data you actually have."""
    cfg = cfg or load_config()
    images = load_train_images(cfg)
    annotations = load_train_annotations(cfg)

    sources = {image.source_id for image in images}
    per_source: dict[str, int] = {}
    for image in images:
        per_source[image.source_id] = per_source.get(image.source_id, 0) + 1

    per_image_counts = [image.n_instances for image in images]
    category_histogram: dict[int, int] = {}
    for annotation in annotations:
        key = annotation["category_id"]
        category_histogram[key] = category_histogram.get(key, 0) + 1

    ignore_regions = load_ignore_regions(cfg)
    ignore_images = {region["image_id"] for region in ignore_regions}

    return {
        "n_images": len(images),
        "n_sources": len(sources),
        "images_per_source_min": min(per_source.values()) if per_source else 0,
        "images_per_source_max": max(per_source.values()) if per_source else 0,
        "n_annotations": len(annotations),
        "submit_category_histogram": dict(sorted(category_histogram.items())),
        "instances_per_image_min": min(per_image_counts) if per_image_counts else 0,
        "instances_per_image_max": max(per_image_counts) if per_image_counts else 0,
        "instances_per_image_mean": (
            round(sum(per_image_counts) / len(per_image_counts), 2) if per_image_counts else 0.0
        ),
        "n_ignore_regions": len(ignore_regions),
        "n_images_with_ignore": len(ignore_images),
        "grouping_looks_broken": len(sources) == len(images) and len(images) > 1,
    }


def write_manifest(path: str | Path, images: Sequence[ImageRecord]) -> Path:
    return write_csv_rows(path, (image.to_row() for image in images), MANIFEST_FIELDS)


def read_manifest(path: str | Path) -> list[ImageRecord]:
    return [
        ImageRecord(
            image_id=int(row["image_id"]),
            file_name=row["file_name"],
            width=int(row["width"]),
            height=int(row["height"]),
            source_id=row["source_id"],
            n_instances=int(row["n_instances"]),
            fold=int(row["fold"]) if str(row.get("fold", "-1")).strip() not in ("", "-1") else -1,
        )
        for row in read_csv_rows(path)
    ]


def main() -> None:  # pragma: no cover - CLI glue
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the COCO training data")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    stats = dataset_stats(cfg)
    images = load_train_images(cfg)
    out = args.out or cfg.path("data.processed_dir") / "manifest.csv"
    write_manifest(out, images)
    stats["manifest"] = str(out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
