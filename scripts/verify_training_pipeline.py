#!/usr/bin/env python
"""End-to-end smoke test of the training pipeline — run this BEFORE a long run.

Why this exists: two bugs reached a full 12-epoch run and were only caught by
scoring the finished model.

1. ``targets["masks"]`` was ``[N, 1, H, W]`` instead of torchvision's required
   ``[N, H, W]``. The box head trained normally; the mask head collapsed to ~0.
2. ``targets["labels"]`` used 0-based submission ids instead of torchvision's
   1-based object labels (0 = background). Class 0 was trained as *background*,
   so the model predicted only one class.

Both are invisible in the total loss. This script instead trains a couple of
images until they are memorised, then runs inference on those *same* images and
asserts the model can actually reproduce them. That exercises the whole chain -
dataset, label numbering, mask rasterisation, loss, and the prediction path -
in a few minutes.

Usage
-----
    python scripts/verify_training_pipeline.py
    python scripts/verify_training_pipeline.py --iters 200 --short-side 512
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.config import load_config  # noqa: E402
from data.coco import load_train_annotations, load_train_images  # noqa: E402
from submit.result_json import encode_mask_to_rle  # noqa: E402


def _load_train_module():
    path = REPO_ROOT / "scripts" / "train_segmentation.py"
    spec = importlib.util.spec_from_file_location("_train_segmentation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def pick_probe_images(images, annotations, per_class: int = 2):
    """Choose a few images that between them cover both classes."""
    by_image: dict[int, set[int]] = {}
    for annotation in annotations:
        by_image.setdefault(annotation["image_id"], set()).add(annotation["category_id"])

    chosen: list = []
    for wanted in (0, 1):
        picked = 0
        for image in images:
            if picked >= per_class:
                break
            if wanted in by_image.get(image.image_id, set()):
                chosen.append(image)
                picked += 1
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--short-side", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        import torchvision
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(f"requires torch + torchvision: {exc}") from exc

    train_module = _load_train_module()
    cfg = load_config()
    device = args.device or train_module.pick_device(torch)

    images = load_train_images(cfg)
    annotations = load_train_annotations(cfg)
    probe = pick_probe_images(images, annotations)
    if not probe:
        raise SystemExit("could not find probe images")
    print(f"device={device}  probe images={len(probe)}  iters={args.iters}")

    dataset = train_module.build_dataset(
        probe, annotations, cfg.path("data.train_images"), args.short_side, False
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=train_module.collate,
    )

    # ---- contract assertions (cheap, catch the two known failure modes) ----
    _, sample = dataset[0]
    num_classes = len(cfg.competition.submit_labels) + 1
    assert sample["masks"].dim() == 3, f"masks must be [N,H,W], got {tuple(sample['masks'].shape)}"
    labels = sorted(set(sample["labels"].tolist()))
    assert min(labels) >= 1, f"label 0 is background; got {labels}"
    assert max(labels) < num_classes, f"labels must be < {num_classes}; got {labels}"
    print(f"  contract OK: masks {tuple(sample['masks'].shape)} labels {labels}")

    model = train_module.build_model(
        torch, torchvision, num_classes, light=True, pretrained=True
    ).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # ---- overfit the probe images ----
    model.train()
    iterator = iter(loader)
    for step in range(1, args.iters + 1):
        try:
            batch_images, batch_targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch_images, batch_targets = next(iterator)
        batch_images = [image.to(device) for image in batch_images]
        batch_targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in batch_targets
        ]
        loss_dict = model(batch_images, batch_targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == args.iters:
            parts = " ".join(f"{k.replace('loss_', '')}={float(v.detach()):.3f}"
                             for k, v in sorted(loss_dict.items()))
            print(f"    iter {step}/{args.iters} loss={float(loss):.4f} [{parts}]", flush=True)

    # ---- the real test: can it reproduce the images it just memorised? ----
    model.eval()
    failures: list[str] = []
    seen_categories: set[int] = set()
    total_instances = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            tensor, target = dataset[index]
            output = model([tensor.to(device)])[0]
            keep = output["scores"] >= 0.5
            n = int(keep.sum())
            total_instances += n
            if n:
                for label in output["labels"][keep].cpu().tolist():
                    seen_categories.add(int(label) - 1)  # back to submission ids
            truth = sorted(set((target["labels"].tolist())))
            print(f"    image {index}: truth labels {truth} -> {n} detections "
                  f"at score>=0.5")

            # masks must survive the full round trip into RLE
            for mask in output["masks"][keep, 0].cpu():
                binary = (mask.numpy() > 0.5).astype("uint8")
                if not binary.any():
                    failures.append(f"image {index}: empty mask for a kept detection")
                    continue
                try:
                    encode_mask_to_rle(binary)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"image {index}: RLE failed ({exc})")

    print()
    print(f"  detections at score>=0.5: {total_instances}")
    print(f"  categories recovered: {sorted(seen_categories)}")

    if total_instances == 0:
        failures.append("model memorised nothing - detection head is not learning")
    if seen_categories != {0, 1}:
        failures.append(
            f"expected both classes after overfitting, saw {sorted(seen_categories)}"
        )
    if failures:
        print("\n❌ pipeline verification FAILED:")
        for failure in failures:
            print(f"   - {failure}")
        return 1

    print("\n✅ pipeline verified: both classes detected and masks encode to valid RLE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
