#!/usr/bin/env python
"""Score a checkpoint on a held-out fold with the frozen official scorer.

This is the only number that may be used to choose a model. Test labels are
withheld, so local evidence has to come from training data split by *source
image* - the fold file already guarantees ``IMG_x.jpg`` and ``IMG_x_aug1.jpg``
are on the same side.

The score comes from ``src/eval/official_oracle/official_evaluate.py``, invoked
as a subprocess, including its ignore-region policy. Nothing here re-implements
the metric.

Usage
-----
    python scripts/evaluate_local.py --fold 0
    python scripts/evaluate_local.py --fold 0 --limit 12     # quick check
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
from data.split_by_source import load_folds  # noqa: E402
from eval.mask_map import final_score, run_official_scorer, summarize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--folds", default=None,
                        help="comma list (e.g. 0,1,2,3,4): score the multi-fold "
                             "ensemble on fold 0's holdout instead of a single "
                             "model; overrides --fold")
    parser.add_argument("--limit", type=int, default=None)
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
        raise SystemExit(f"evaluation requires torch: {exc}") from exc

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from predict_test import load_models, parse_short_sides, pick_device, predict_image

    cfg = load_config()
    device = args.device or pick_device(torch)

    if args.folds:
        eval_folds = [int(v) for v in args.folds.split(",") if v.strip()]
        work_tag = f"ensemble_fold{eval_folds[0]}"
        eval_fold = eval_folds[0]
    else:
        if args.fold is None:
            raise SystemExit("either --fold N or --folds a,b,c is required")
        eval_folds = [args.fold]
        work_tag = f"fold{args.fold}"
        eval_fold = args.fold

    # Keep resolution sweeps in their own work dir: reusing the tag would
    # overwrite the 640px predictions and GT that earlier numbers came from.
    if args.short_side:
        tag = "-".join(str(v) for v in parse_short_sides(args.short_side))
        work_tag = f"{work_tag}_s{tag}"

    folds_path = cfg.path("split.output_dir") / "folds.json"
    if not folds_path.is_file():
        raise SystemExit(f"folds not found: {folds_path}. Run: make split")
    split, images = load_folds(folds_path)

    fold = next((f for f in split.folds if f.fold == eval_fold), None)
    if fold is None:
        raise SystemExit(f"fold {eval_fold} not in {[f.fold for f in split.folds]}")

    by_id = {image.image_id: image for image in images}
    val_ids = list(fold.val_image_ids)
    if args.limit:
        val_ids = val_ids[: args.limit]
    val_images = [by_id[image_id] for image_id in val_ids]

    work_dir = cfg.path("eval.output_dir") / work_tag
    work_dir.mkdir(parents=True, exist_ok=True)
    image_dir = cfg.path("data.train_images")

    print(f"device={device}  models=folds{eval_folds}  val images={len(val_images)}")
    models = load_models(cfg, eval_folds, device, short_side=args.short_side)

    # Predictions are written in submission numbering (0/1); the official scorer
    # maps the GT's 1/2 itself, so no conversion happens here.
    results = []
    started = time.time()
    for index, image in enumerate(val_images, 1):
        instances = predict_image(models, image_dir / image.file_name, cfg, device)
        results.append({"image_id": image.file_name, "instances": instances})
        if index % 10 == 0 or index == len(val_images):
            print(f"  {index}/{len(val_images)}  ({time.time() - started:.0f}s)")

    submission = work_dir / "result.json"
    submission.write_text(
        json.dumps({"version": "1.0", "results": results}, ensure_ascii=False),
        encoding="utf-8",
    )

    # The fold GT is built from the *training* annotations restricted to this
    # fold's validation images - the only labelled data that exists locally.
    from eval.mask_map import build_fold_ground_truth

    gt_json, ignore_json = build_fold_ground_truth(
        cfg, val_ids, work_dir / "gt.json", work_dir / "ignore.json"
    )
    payload = run_official_scorer(
        cfg,
        submission_json=submission,
        gt_json=gt_json,
        ignore_json=ignore_json,
        output_json=work_dir / "score.json",
        error_json=work_dir / "error.json",
    )

    summary = summarize(payload)
    print("\n" + "=" * 64)
    print(f"folds{eval_folds}  Mask mAP@[0.50:0.95] = {final_score(payload):.6f}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 64)

    out = args.out or (cfg.path("eval.output_dir") / f"{work_tag}_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
