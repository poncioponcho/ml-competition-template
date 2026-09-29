"""Grouped K-fold split keyed on the *source* image.

``sample.jpg`` and ``sample_aug1.jpg`` are the same photograph, one of them
flipped. Splitting them across train/val would leak near-identical content and
inflate the local Mask mAP, which is the only signal available for model
selection (test labels are withheld). The grouping is therefore enforced by a
hard assertion rather than left to convention.

Determinism: the fold assignment is a pure function of
``(source ids, n_folds, seed)``, so any reported local score is reproducible.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from common.config import Config, load_config
from common.io_utils import write_json
from data.coco import ImageRecord, load_train_images


class SplitError(ValueError):
    """Raised when a split violates the grouping or coverage contract."""


@dataclass(frozen=True)
class Fold:
    fold: int
    train_image_ids: tuple[int, ...]
    val_image_ids: tuple[int, ...]
    train_sources: tuple[str, ...]
    val_sources: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ("train_image_ids", "val_image_ids", "train_sources", "val_sources"):
            payload[key] = list(payload[key])
        return payload


@dataclass
class SplitResult:
    n_folds: int
    seed: int
    folds: list[Fold]
    audit: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": "grouped_kfold_by_source",
            "n_folds": self.n_folds,
            "seed": self.seed,
            "folds": [fold.to_dict() for fold in self.folds],
            "audit": self.audit,
        }


def assign_sources_to_folds(
    source_ids: Sequence[str], *, n_folds: int, seed: int
) -> dict[str, int]:
    """Deterministically map each source image to a fold (LPT balancing)."""
    unique = sorted(set(source_ids))
    if n_folds < 2:
        raise SplitError(f"n_folds must be >= 2, got {n_folds}")
    if len(unique) < n_folds:
        raise SplitError(f"need at least {n_folds} sources, got {len(unique)}")

    rng = random.Random(seed)
    shuffled = unique[:]
    rng.shuffle(shuffled)

    loads = [0] * n_folds
    assignment: dict[str, int] = {}
    for source in shuffled:
        target = min(range(n_folds), key=lambda f: (loads[f], f))
        assignment[source] = target
        loads[target] += 1
    return assignment


def assert_no_source_leakage(folds: Iterable[Fold]) -> None:
    """Hard gate: a source image may not appear on both sides of a fold."""
    for fold in folds:
        overlap = set(fold.train_sources) & set(fold.val_sources)
        if overlap:
            raise SplitError(
                f"source leakage in fold {fold.fold}: {sorted(overlap)[:5]} "
                f"({len(overlap)} sources)"
            )


def assert_full_coverage(folds: Iterable[Fold], images: Sequence[ImageRecord]) -> None:
    """Every image is validated exactly once across all folds."""
    counter: Counter[int] = Counter()
    for fold in folds:
        counter.update(fold.val_image_ids)
    expected = {image.image_id for image in images}
    seen = set(counter)
    if seen != expected:
        raise SplitError(
            f"validation coverage mismatch: missing={sorted(expected - seen)[:5]} "
            f"extra={sorted(seen - expected)[:5]}"
        )
    duplicated = sorted(image_id for image_id, count in counter.items() if count != 1)
    if duplicated:
        raise SplitError(f"images in more than one val fold: {duplicated[:5]}")


def build_folds(images: Sequence[ImageRecord], *, n_folds: int, seed: int) -> SplitResult:
    """Grouped K-fold with all leakage/coverage assertions applied."""
    images = list(images)
    if not images:
        raise SplitError("no images to split")

    assignment = assign_sources_to_folds(
        [image.source_id for image in images], n_folds=n_folds, seed=seed
    )

    by_fold: dict[int, list[int]] = {fold_id: [] for fold_id in range(n_folds)}
    for image in images:
        by_fold[assignment[image.source_id]].append(image.image_id)

    folds: list[Fold] = []
    all_ids = {image.image_id for image in images}
    for fold_id in range(n_folds):
        val_ids = tuple(sorted(by_fold[fold_id]))
        if not val_ids:
            raise SplitError(f"fold {fold_id} is empty; reduce n_folds")
        val_set = set(val_ids)
        train_ids = tuple(sorted(all_ids - val_set))
        folds.append(
            Fold(
                fold=fold_id,
                train_image_ids=train_ids,
                val_image_ids=val_ids,
                train_sources=tuple(sorted({image.source_id for image in images
                                            if image.image_id in set(train_ids)})),
                val_sources=tuple(sorted({image.source_id for image in images
                                          if image.image_id in val_set})),
            )
        )

    assert_no_source_leakage(folds)
    assert_full_coverage(folds, images)

    audit = {
        "n_images": len(images),
        "n_sources": len({image.source_id for image in images}),
        "fold_val_image_counts": [len(fold.val_image_ids) for fold in folds],
        "fold_val_source_counts": [len(fold.val_sources) for fold in folds],
    }
    return SplitResult(n_folds=n_folds, seed=seed, folds=folds, audit=audit)


def fold_summary(result: SplitResult) -> str:
    lines = [
        f"grouped {result.n_folds}-fold by source image  (seed={result.seed})",
        f"{'fold':>4}  {'#train':>7}  {'#val':>6}  {'#src_train':>10}  {'#src_val':>9}",
    ]
    for fold in result.folds:
        lines.append(
            f"{fold.fold:>4}  {len(fold.train_image_ids):>7}  {len(fold.val_image_ids):>6}  "
            f"{len(fold.train_sources):>10}  {len(fold.val_sources):>9}"
        )
    return "\n".join(lines)


def run_split(
    cfg: Config | None = None,
    *,
    n_folds: int | None = None,
    seed: int | None = None,
) -> SplitResult:
    cfg = cfg or load_config()
    n_folds = n_folds if n_folds is not None else int(cfg.split.n_folds)
    seed = seed if seed is not None else int(cfg.split.seed)

    images = load_train_images(cfg)
    result = build_folds(images, n_folds=n_folds, seed=seed)

    out_dir = cfg.path("split.output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["images"] = [image.to_row() for image in images]
    write_json(out_dir / "folds.json", payload)
    (out_dir / "summary.txt").write_text(fold_summary(result) + "\n", encoding="utf-8")
    return result


def load_folds(path: str | Path) -> tuple[SplitResult, list[ImageRecord]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = [
        Fold(
            fold=entry["fold"],
            train_image_ids=tuple(entry["train_image_ids"]),
            val_image_ids=tuple(entry["val_image_ids"]),
            train_sources=tuple(entry["train_sources"]),
            val_sources=tuple(entry["val_sources"]),
        )
        for entry in raw["folds"]
    ]
    images = [
        ImageRecord(
            image_id=int(row["image_id"]),
            file_name=row["file_name"],
            width=int(row["width"]),
            height=int(row["height"]),
            source_id=row["source_id"],
            n_instances=int(row["n_instances"]),
            fold=int(row.get("fold", -1)),
        )
        for row in raw["images"]
    ]
    return (
        SplitResult(raw["n_folds"], raw["seed"], folds, raw.get("audit", {})),
        images,
    )


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    result = run_split(cfg, n_folds=args.n_folds, seed=args.seed)
    print(fold_summary(result))
    print(json.dumps(result.audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
