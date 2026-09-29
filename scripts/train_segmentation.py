#!/usr/bin/env python
"""Train an instance-segmentation model with source-grouped folds.

Design notes tied to this competition:

* **Grouped folds.** The fold file comes from ``src/data/split_by_source.py``,
  which already asserted that ``IMG_x.jpg`` and ``IMG_x_aug1.jpg`` never land on
  opposite sides. Test labels are withheld, so a grouped local Mask mAP is the
  only evidence available for model selection.
* **Category renumbering.** Targets are built in *submission* numbering (0/1) by
  ``data.coco``; the local evaluator maps back to GT numbering when scoring.
* **Ignore regions are not trained on.** They are a scoring device, not a class.
  Instances overlapping them are still valid training targets, so nothing is
  masked out during training.
* **Resolution.** Source images are 2048x1152. Training at full size is
  expensive, so the short side is scaled (config ``train.train_short_side``).
  Predictions are resized back to the original size before RLE encoding, as the
  official instructions require.

Usage
-----
    python scripts/train_segmentation.py --folds 0
    python scripts/train_segmentation.py --light --epochs 3      # quick smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.config import load_config  # noqa: E402
from data.coco import load_train_annotations, load_train_images  # noqa: E402
from data.split_by_source import load_folds  # noqa: E402


def _require_torch():
    try:
        import torch
        import torchvision
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"training requires torch + torchvision: {exc}\n"
            "install with: pip install torch torchvision"
        ) from exc
    return torch, torchvision


def pick_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_scheduler(torch, optimizer, epochs: int, start_epoch: int, base_lr: float):
    """Cosine schedule positioned so that epoch ``start_epoch`` gets a real lr.

    Resuming is a trap here. ``optimizer.load_state_dict()`` restores the whole
    param group, **including ``lr``**. A cosine schedule that has run to the end
    of its budget stores ``lr = 0``, so a resumed run happily trains its entire
    remaining budget at lr = 0: the loss curve looks plausible (batch-to-batch
    noise), the log prints every step, and the saved weights are bit-identical
    to the checkpoint it started from. That is exactly what happened to fold 0
    on 2026-09-28 - 12 epochs (~2.9 h) of training changed nothing at all.

    Rebuilding the schedule from ``base_lr`` and fast-forwarding it to
    ``start_epoch`` is what makes the remaining epochs actually train - and
    fast-forwarding (rather than passing ``last_epoch``) keeps the resumed run on
    the same curve it would have followed had it never been interrupted, because
    cosine annealing's recursion telescopes to the closed form.
    """
    for group in optimizer.param_groups:
        group["initial_lr"] = base_lr
        group["lr"] = base_lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    # Fast-forwarding steps the scheduler without stepping the optimizer, which
    # torch warns about. It is intentional here: no weights are touched, only the
    # schedule is advanced to the point the run had reached.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*"
        )
        for _ in range(start_epoch):
            scheduler.step()
    return scheduler


def build_dataset(records, annotations, image_dir: Path, short_side: int, train: bool):
    """Dataset returning ``(image_tensor, target)`` with masks from COCO polygons.

    ``CocoDataset`` is defined at module scope on purpose: a class defined
    inside a function cannot be pickled, which breaks every DataLoader with
    ``num_workers > 0``.
    """
    by_image: dict[int, list[dict]] = {}
    for annotation in annotations:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
    return CocoDataset(list(records), by_image, Path(image_dir), short_side, train)


class CocoDataset:
    """Yields ``(image_tensor, target)`` for one image, with polygon->mask decoding.

    Caches the *pre-augmentation* resized image and its masks in memory. Source
    JPEGs are 2048x1152 and there are only ~700 of them, so decoding them once
    per epoch is pure overhead: at 640px the whole cache is a few hundred MB,
    and every epoch after the first becomes compute-bound instead of IO-bound.
    """

    def __init__(self, records, by_image, image_dir: Path, short_side: int, train: bool):
        self.records = list(records)
        self.by_image = by_image
        self.image_dir = Path(image_dir)
        self.short_side = int(short_side)
        self.train = bool(train)
        self._cache: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, index: int):
        """Decode, resize and rasterise once; returns (image_array, masks, labels, boxes)."""
        import numpy as np
        from PIL import Image
        from pycocotools import mask as mask_utils

        cached = self._cache.get(index)
        if cached is not None:
            return cached

        record = self.records[index]
        with Image.open(self.image_dir / record.file_name) as handle:
            image = handle.convert("RGB")
        width, height = image.size
        scale = self.short_side / min(width, height)
        new_w, new_h = int(round(width * scale)), int(round(height * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)

        masks, labels, boxes = [], [], []
        for annotation in self.by_image.get(record.image_id, []):
            rle = mask_utils.frPyObjects(annotation["segmentation"], height, width)
            merged = mask_utils.merge(rle) if isinstance(rle, list) else rle
            mask = mask_utils.decode(merged)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0).astype(np.uint8)
            if not mask.any():
                continue
            resized = np.array(
                Image.fromarray(mask * 255).resize((new_w, new_h), Image.NEAREST)
            ) > 127
            ys, xs = np.nonzero(resized)
            if ys.size == 0:
                continue
            masks.append(resized)
            # torchvision reserves label 0 for BACKGROUND; object labels must be
            # 1..num_classes-1. Submission ids are 0-based, so shift by one here
            # and subtract it back at prediction time. Feeding 0-based ids makes
            # the model train class 0 as background, and it then predicts only
            # one class - a failure that is invisible in the loss curve.
            labels.append(int(annotation["category_id"]) + 1)
            boxes.append(
                [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            )

        payload = (np.array(image), masks, labels, boxes)
        self._cache[index] = payload
        return payload

    def __getitem__(self, index: int):
        import random

        import numpy as np
        import torch
        import torchvision.transforms.functional as TF

        image_array, masks, labels, boxes = self._load(index)
        image = TF.to_pil_image(image_array)
        new_h, new_w = image_array.shape[0], image_array.shape[1]

        flip = self.train and random.random() < 0.5
        if flip:
            image = TF.hflip(image)
        if self.train:
            # Gentle photometric jitter: colour is a real cue here, so heavy
            # jitter would destroy the very signal the model must learn.
            image = TF.adjust_brightness(image, 1.0 + random.uniform(-0.15, 0.15))
            image = TF.adjust_contrast(image, 1.0 + random.uniform(-0.15, 0.15))
            image = TF.adjust_saturation(image, 1.0 + random.uniform(-0.12, 0.12))

        tensor = TF.to_tensor(image)

        final_masks, final_boxes = [], []
        for mask, box in zip(masks, boxes):
            if flip:
                mask = mask[:, ::-1]
                x0, y0, x1, y1 = box
                box = [new_w - x1, y0, new_w - x0, y1]
            if not mask.any():
                continue
            final_masks.append(np.ascontiguousarray(mask))
            final_boxes.append(box)

        # torchvision expects targets["masks"] as UInt8Tensor[N, H, W] - NOT
        # [N, 1, H, W]. Getting this wrong trains the box head normally while
        # the mask head collapses to ~0, which is invisible in the total loss.
        mask_targets = (
            np.stack(final_masks) if final_masks else np.zeros((0, new_h, new_w))
        )
        target = {
            "boxes": torch.as_tensor(final_boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels[: len(final_boxes)], dtype=torch.int64),
            "masks": torch.as_tensor(mask_targets, dtype=torch.uint8),
            "image_id": torch.tensor([self.records[index].image_id]),
            "area": torch.as_tensor(
                [float(m.sum()) for m in final_masks], dtype=torch.float32
            ),
            "iscrowd": torch.zeros((len(final_masks),), dtype=torch.int64),
        }
        return tensor, target


def build_model(torch, torchvision, num_classes: int, *, light: bool, pretrained: bool):
    """Mask R-CNN in submission numbering: 0 = background, 1..2 = classes 0/1."""
    weights = None
    if pretrained:
        weights = (
            torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            if not light
            else torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        )
    factory = (
        torchvision.models.detection.maskrcnn_resnet50_fpn
        if light
        else torchvision.models.detection.maskrcnn_resnet50_fpn_v2
    )
    model = factory(weights=weights, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    hidden = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(hidden, hidden, num_classes)
    return model


def collate(batch):
    return tuple(zip(*batch))


def train_fold(cfg, images, annotations, fold, args, device):
    import torch
    from torch.utils.data import DataLoader

    torch, torchvision = _require_torch()
    image_dir = cfg.path("data.train_images")
    short_side = int(args.short_side or cfg.train.train_short_side)
    epochs = int(args.epochs or cfg.train.epochs)
    batch_size = int(args.batch_size or cfg.train.batch_size)

    train_records = [i for i in images if i.image_id in set(fold.train_image_ids)]
    val_records = [i for i in images if i.image_id in set(fold.val_image_ids)]
    print(f"  fold {fold.fold}: {len(train_records)} train / {len(val_records)} val images "
          f"@ short_side={short_side}")

    train_ds = build_dataset(train_records, annotations, image_dir, short_side, True)

    # Fail fast on the target contract. A malformed mask target (e.g. [N,1,H,W]
    # instead of [N,H,W]) trains the box head normally while the mask head
    # collapses to ~0, and the total loss gives no hint that anything is wrong.
    _, probe = train_ds[0]
    assert probe["masks"].dim() == 3, (
        "targets['masks'] must be UInt8Tensor[N, H, W]; got "
        f"{tuple(probe['masks'].shape)}"
    )
    assert probe["masks"].shape[0] == probe["boxes"].shape[0], (
        "masks/boxes instance count mismatch: "
        f"{probe['masks'].shape[0]} vs {probe['boxes'].shape[0]}"
    )
    if probe["masks"].shape[0]:
        assert bool(probe["masks"].any()), "mask target is all zeros"
    # Label 0 is reserved for background by torchvision. Feeding 0-based
    # submission ids trains class 0 as background and the model then predicts
    # only the other class - visible in per-class AP, invisible in the loss.
    num_classes = len(cfg.competition.submit_labels) + 1
    label_values = sorted(set(probe["labels"].tolist()))
    assert label_values and min(label_values) >= 1 and max(label_values) < num_classes, (
        "targets['labels'] must lie in 1..num_classes-1 (0 = background); got "
        f"{label_values} with num_classes={num_classes}"
    )
    print(f"  target contract OK: masks {tuple(probe['masks'].shape)}, "
          f"boxes {tuple(probe['boxes'].shape)}, labels {probe['labels'].tolist()} "
          f"(num_classes={num_classes})", flush=True)

    # Default to 0 workers: the dataset caches decoded+resized images in memory,
    # so a single process avoids duplicating that cache per worker (and avoids
    # spawn overhead entirely). Raise it only if profiling says otherwise.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=int(args.num_workers), collate_fn=collate,
    )

    model = build_model(
        torch, torchvision, len(cfg.competition.submit_labels) + 1,
        light=args.light, pretrained=not args.no_pretrained,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=float(cfg.train.lr), momentum=float(cfg.train.momentum),
        weight_decay=float(cfg.train.weight_decay),
    )
    base_lr = float(cfg.train.lr)

    out_dir = cfg.path("train.output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"fold{fold.fold}.pt"

    # Resume support: long runs get killed by the host, so the same command must
    # be able to pick up where it stopped instead of losing hours of work.
    start_epoch = 0
    if not args.no_resume and checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        same_setup = (
            int(payload.get("short_side", -1)) == short_side
            and bool(payload.get("light", False)) == bool(args.light)
        )
        if same_setup:
            model.load_state_dict(payload["model"])
            if payload.get("optimizer"):
                optimizer.load_state_dict(payload["optimizer"])
            start_epoch = int(payload.get("epoch", 0))
            print(f"  fold {fold.fold}: RESUMED from {checkpoint.name} "
                  f"at epoch {start_epoch}/{epochs}", flush=True)
        else:
            print(f"  fold {fold.fold}: existing checkpoint has a different setup "
                  f"(short_side={payload.get('short_side')}, light={payload.get('light')}); "
                  f"starting fresh", flush=True)
    if start_epoch >= epochs:
        print(f"  fold {fold.fold}: already trained {start_epoch} epochs, nothing to do",
              flush=True)
        return checkpoint

    # The scheduler is built *after* the optimizer state is restored, so the
    # resumed run cannot inherit an annealed-to-zero learning rate.
    scheduler = make_scheduler(torch, optimizer, epochs, start_epoch, base_lr)
    resumed_lr = optimizer.param_groups[0]["lr"]
    if not any(group["lr"] > 0 for group in optimizer.param_groups):
        raise SystemExit(
            f"fold {fold.fold}: refusing to train with lr=0 - epochs "
            f"{start_epoch + 1}..{epochs} would be silent no-ops"
        )
    print(f"  fold {fold.fold}: lr for epoch {start_epoch + 1} = {resumed_lr:.6g}",
          flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        started = time.time()
        running = 0.0
        running_parts: dict[str, float] = {}
        for step, (images_batch, targets) in enumerate(train_loader, 1):
            images_batch = [image.to(device) for image in images_batch]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            loss_dict = model(images_batch, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(cfg.train.grad_clip))
            optimizer.step()
            running += float(loss.detach())
            for key, value in loss_dict.items():
                running_parts[key] = running_parts.get(key, 0.0) + float(value.detach())
            if step % 20 == 0:
                elapsed = time.time() - started
                rate = step / elapsed if elapsed else 0
                eta = (len(train_loader) - step) / rate if rate else 0
                parts = " ".join(
                    f"{k.replace('loss_', '')}={v / step:.3f}"
                    for k, v in sorted(running_parts.items())
                )
                print(f"    epoch {epoch + 1}/{epochs} step {step}/{len(train_loader)} "
                      f"loss={running / step:.4f} [{parts}] {rate:.2f} it/s "
                      f"eta {eta / 60:.1f} min", flush=True)
        scheduler.step()
        mean_parts = " ".join(
            f"{k.replace('loss_', '')}={v / max(1, len(train_loader)):.4f}"
            for k, v in sorted(running_parts.items())
        )
        print(f"  fold {fold.fold} epoch {epoch + 1}/{epochs} done in "
              f"{time.time() - started:.0f}s, mean loss "
              f"{running / max(1, len(train_loader)):.4f} [{mean_parts}]", flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "fold": fold.fold,
                "light": bool(args.light),
                "short_side": short_side,
                "num_classes": len(cfg.competition.submit_labels) + 1,
            },
            checkpoint,
        )
        print(f"  fold {fold.fold} checkpoint saved -> {checkpoint.name} "
              f"(epoch {epoch + 1})", flush=True)
    print(f"  fold {fold.fold} checkpoint -> {checkpoint}")
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default=None, help="e.g. 0,1 or omit for all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--short-side", type=int, default=None)
    parser.add_argument("--light", action="store_true",
                        help="use maskrcnn_resnet50_fpn (v1) as a faster baseline")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an existing checkpoint and start fresh")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch, _ = _require_torch()
    device = args.device or pick_device(torch)
    cfg = load_config()

    folds_path = cfg.path("split.output_dir") / "folds.json"
    if not folds_path.is_file():
        raise SystemExit(f"folds not found: {folds_path}\nRun: make split")

    split, images = load_folds(folds_path)
    annotations = load_train_annotations(cfg)
    requested = (
        [int(v) for v in args.folds.split(",")] if args.folds
        else list(range(split.n_folds))
    )
    print(f"device={device}  folds={requested}  images={len(images)}  "
          f"annotations={len(annotations)}")

    for fold in split.folds:
        if fold.fold in requested:
            train_fold(cfg, images, annotations, fold, args, device)

    print("\ntraining complete. Next:")
    print("  make predict          # inference on testB")
    print("  make submission       # build + verify b_submission.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
