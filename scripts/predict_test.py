#!/usr/bin/env python
"""Run inference on a test split and write raw predictions.

Output format (consumed by ``scripts/build_submission.py``)::

    {"sample.jpg": [{"category_id": 0, "score": 0.91,
                      "segmentation": {"size": [1152, 2048], "counts": "..."}}]}

Masks are predicted at a reduced resolution and **resized back to the original
image size before RLE encoding**, which the official instructions require
("必须先把预测掩码恢复至原图分辨率，再编码为真实 COCO compressed RLE").

Multi-fold checkpoints are ensembled by per-class NMS over the union of their
detections, then truncated to ``max_detections_per_image`` (the official
maxDets cap).

Usage
-----
    python scripts/predict_test.py --split testB
    python scripts/predict_test.py --split testB --folds 0 --limit 8   # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.config import load_config  # noqa: E402
from submit.result_json import encode_mask_to_rle, load_test_manifest  # noqa: E402


def pick_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_short_sides(value) -> list[int] | None:
    """Accept ``1024`` or ``640,1024`` and return a list of resolutions."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        return None
    return [int(part) for part in parts]


def load_models(cfg, folds, device, *, light: bool | None = None,
                short_side=None):
    """Load one model per fold checkpoint found on disk.

    ``folds=None`` means "every checkpoint present", which is the ensemble case.

    ``short_side`` overrides the resolution recorded in the checkpoints and may
    list several (``640,1024``): each model is then run at every listed scale and
    the detections are fused, which is multi-scale TTA. The checkpoints hold the
    *training* short side, which is a poor default for inference - the masks are
    what the metric scores, and mask boundaries sharpen as the input grows.
    """
    import torch
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    resolutions = parse_short_sides(short_side)
    out_dir = cfg.path("train.output_dir")
    if folds is None:
        folds = sorted(
            int(path.stem[len("fold"):]) for path in out_dir.glob("fold*.pt")
            if path.stem[len("fold"):].isdigit()
        )
    checkpoints = [
        out_dir / f"fold{fold}.pt" for fold in folds
        if (out_dir / f"fold{fold}.pt").is_file()
    ]
    if not checkpoints:
        raise SystemExit(
            f"no fold checkpoints under {out_dir}. Train first (make train)."
        )

    models = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        use_light = bool(payload.get("light", False)) if light is None else light
        factory = (
            torchvision.models.detection.maskrcnn_resnet50_fpn
            if use_light
            else torchvision.models.detection.maskrcnn_resnet50_fpn_v2
        )
        model = factory(weights=None, weights_backbone=None)
        num_classes = int(payload.get("num_classes", 3))
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        hidden = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(hidden, hidden, num_classes)
        model.load_state_dict(payload["model"])
        model.eval()
        model.to(device)
        trained_at = int(payload.get("short_side", 1024))
        infer_at = resolutions if resolutions else [trained_at]
        models.append((checkpoint.name, model, infer_at))
        note = "" if infer_at == [trained_at] else f" (trained at {trained_at})"
        print(f"  loaded {checkpoint.name} (infer short_side={infer_at}{note})")
    return models


def predict_image(models, image_path: Path, cfg, device) -> list[dict]:
    """Run every (fold, resolution) member on one image and fuse the detections."""
    import numpy as np
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torchvision.ops import nms

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    width, height = image.size

    boxes_all, scores_all, labels_all, masks_all = [], [], [], []
    for _, model, resolutions in models:
        for short_side in resolutions:
            scale = short_side / min(width, height)
            new_w, new_h = int(round(width * scale)), int(round(height * scale))
            resized = image.resize((new_w, new_h), Image.BILINEAR)
            tensor = TF.to_tensor(resized).to(device)

            with torch.no_grad():
                output = model([tensor])[0]

            keep = output["scores"] >= float(cfg.inference.score_threshold)
            if not bool(keep.any()):
                continue
            boxes = output["boxes"][keep].cpu()
            scores = output["scores"][keep].cpu()
            labels = output["labels"][keep].cpu()
            masks = output["masks"][keep, 0].cpu()

            # Undo the inference resize so masks land on the original grid.
            boxes = boxes * torch.tensor(
                [width / new_w, height / new_h, width / new_w, height / new_h]
            )
            masks = torch.stack(
                [
                    torch.from_numpy(
                        np.array(
                            Image.fromarray(
                                (mask.numpy() > 0.5).astype(np.uint8) * 255
                            ).resize((width, height), Image.NEAREST)
                        )
                        > 127
                    )
                    for mask in masks
                ]
            ) if len(masks) else torch.zeros((0, height, width), dtype=torch.bool)

            boxes_all.append(boxes)
            scores_all.append(scores)
            labels_all.append(labels)
            masks_all.append(masks)

    if not boxes_all:
        return []

    boxes = torch.cat(boxes_all)
    scores = torch.cat(scores_all)
    labels = torch.cat(labels_all)
    masks = torch.cat(masks_all)

    # Per-class NMS across folds; the class axis is what matters, since
    # Different classes overlap heavily and must not suppress each other.
    keep_indices: list[int] = []
    for label in labels.unique():
        index = torch.nonzero(labels == label).flatten()
        kept = nms(boxes[index], scores[index], 0.5)
        keep_indices.extend(index[kept].tolist())
    order = sorted(keep_indices, key=lambda i: -float(scores[i]))
    limit = int(cfg.inference.max_detections_per_image)
    order = order[:limit]

    instances = []
    for index in order:
        mask = masks[index].numpy().astype("uint8")
        if not mask.any():
            continue
        try:
            rle = encode_mask_to_rle(mask)
        except Exception:
            continue
        instances.append(
            {
                "category_id": int(labels[index]) - 1,  # back to submission numbering
                "score": round(float(scores[index]), 6),
                "segmentation": rle,
            }
        )
    return instances


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["testB", "testA"], default="testB")
    parser.add_argument("--folds", default=None, help="subset of fold indices")
    parser.add_argument("--limit", type=int, default=None, help="only first N images")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--short-side", default=None,
                        help="inference resolution(s), e.g. 1024 or 640,1024 "
                             "(multiple = multi-scale TTA); default: the short "
                             "side the checkpoint was trained at")
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"inference requires torch: {exc}") from exc

    cfg = load_config()
    device = args.device or pick_device(torch)
    folds = [int(v) for v in args.folds.split(",")] if args.folds else None

    is_b = args.split == "testB"
    manifest = cfg.path("data.test_b_manifest" if is_b else "data.test_a_manifest")
    image_dir = cfg.path("data.test_b_images" if is_b else "data.test_a_images")
    images = load_test_manifest(manifest)
    if args.limit:
        images = images[: args.limit]

    out_path = args.out or (
        cfg.path("inference.raw_predictions_dir") / f"raw_predictions_{args.split}.json"
    )

    print(f"device={device}  split={args.split}  images={len(images)}")
    models = load_models(cfg, folds, device, short_side=args.short_side)

    predictions: dict[str, list[dict]] = {}
    started = time.time()
    for index, image in enumerate(images, 1):
        path = image_dir / image.image_id
        if not path.is_file():
            raise SystemExit(f"test image missing: {path}")
        predictions[image.image_id] = predict_image(models, path, cfg, device)
        if index % 10 == 0 or index == len(images):
            elapsed = time.time() - started
            rate = index / elapsed if elapsed else 0
            remaining = (len(images) - index) / rate if rate else 0
            print(f"  {index}/{len(images)}  {rate:.2f} img/s  "
                  f"eta {remaining / 60:.1f} min")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(predictions, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    total = sum(len(v) for v in predictions.values())
    empty = sum(1 for v in predictions.values() if not v)
    print(f"\nwrote {out_path}")
    print(f"  {len(predictions)} images, {total} instances, {empty} images with no detection")
    print(f"  next: python scripts/build_submission.py --predictions {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
