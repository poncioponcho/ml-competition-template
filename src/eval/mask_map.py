"""Mask mAP evaluation, delegated to the frozen official scoring script.

Architecture (the pattern carried over from the team's previous competition, and
the one lesson worth keeping): the *official* scorer is the only authority. It is
copied byte-for-byte into ``src/eval/official_oracle/``, its SHA-256 is recorded
in ``src/official_freeze.json``, and it is invoked as a subprocess so it never
imports project code. Any edit to it turns a unit test red.

A local, fast pycocotools evaluation is provided for iteration only. It is
*not* ignore-aware, so it reads slightly differently from the official number -
it exists to rank experiments quickly, never to produce a reported score.

Test labels are withheld, so every local number comes from scoring the training
data on held-out source-grouped folds.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

from common.config import Config, load_config
from common.io_utils import write_json
from data.coco import load_coco, load_ignore_regions

DEFAULT_FROZEN_GT_IMAGES_KEY = "images"


class EvalError(RuntimeError):
    """Raised when the official scorer rejects a submission or crashes."""


# --------------------------------------------------------------------------- #
# ground truth for a validation fold
# --------------------------------------------------------------------------- #
def build_fold_ground_truth(
    cfg: Config,
    val_image_ids: Iterable[int],
    out_gt: str | Path,
    out_ignore: str | Path,
) -> tuple[Path, Path]:
    """Write GT + ignore-region JSON restricted to one validation fold.

    The GT keeps its **native** COCO category ids (1 and 2): the official scorer
    performs the 1/2 -> 0/1 correspondence itself. Renumbering here would
    double-map and silently corrupt the score.
    """
    val_ids = set(int(value) for value in val_image_ids)
    if not val_ids:
        raise EvalError("empty validation fold")

    gt_data = load_coco(cfg.path("data.instances_json"))
    gt_images = [image for image in gt_data["images"] if int(image["id"]) in val_ids]
    if len(gt_images) != len(val_ids):
        found = {int(image["id"]) for image in gt_images}
        raise EvalError(
            f"fold images missing from GT: {sorted(val_ids - found)[:5]}"
        )
    gt_annotations = [
        annotation
        for annotation in gt_data["annotations"]
        if int(annotation["image_id"]) in val_ids
    ]
    out_gt = Path(out_gt)
    write_json(
        out_gt,
        {
            "info": gt_data.get("info", {}),
            "licenses": gt_data.get("licenses", []),
            "images": gt_images,
            "annotations": gt_annotations,
            "categories": gt_data["categories"],
        },
    )

    ignore_data_path = cfg.path("data.ignore_json")
    out_ignore = Path(out_ignore)
    if Path(ignore_data_path).is_file():
        ignore_data = load_coco(ignore_data_path)
        keep = [
            region
            for region in ignore_data.get("ignore_regions", [])
            if int(region["image_id"]) in val_ids
        ]
        write_json(
            out_ignore,
            {
                "version": ignore_data.get("version", "1.0"),
                "split": ignore_data.get("split", "train"),
                "description": ignore_data.get(
                    "description", "Category-independent ignore regions."
                ),
                "overlap_metric": ignore_data.get("overlap_metric", "intersection_over_prediction"),
                "overlap_threshold": ignore_data.get("overlap_threshold", 0.5),
                "images": [image for image in ignore_data.get("images", [])
                           if int(image["id"]) in val_ids],
                "ignore_regions": keep,
            },
        )
    else:
        write_json(out_ignore, {"version": "1.0", "images": [], "ignore_regions": []})

    return out_gt, out_ignore


# --------------------------------------------------------------------------- #
# official scorer (frozen, subprocess)
# --------------------------------------------------------------------------- #
def run_official_scorer(
    cfg: Config,
    *,
    submission_json: str | Path,
    gt_json: str | Path,
    ignore_json: str | Path,
    output_json: str | Path,
    error_json: str | Path | None = None,
    python_executable: str | None = None,
    timeout: int = 1800,
) -> dict:
    """Invoke the frozen official scorer and return its parsed ``score.json``."""
    script = cfg.path("eval.oracle_script")
    if not Path(script).is_file():
        raise EvalError(f"frozen official scorer not found: {script}")

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    error_json = Path(error_json) if error_json else output_json.with_suffix(".error.json")

    command = [
        python_executable or sys.executable,
        str(script),
        "--gt_json", str(Path(gt_json).resolve()),
        "--ignore_json", str(Path(ignore_json).resolve()),
        "--submission_json", str(Path(submission_json).resolve()),
        "--output_json", str(output_json.resolve()),
        "--error_json", str(error_json.resolve()),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout
    )
    if completed.returncode != 0 or not output_json.is_file():
        detail = ""
        if error_json.is_file():
            try:
                detail = json.dumps(
                    json.loads(error_json.read_text(encoding="utf-8")), ensure_ascii=False
                )[:2000]
            except Exception:  # pragma: no cover - defensive
                detail = error_json.read_text(encoding="utf-8", errors="replace")[:2000]
        raise EvalError(
            "official scorer failed "
            f"(exit {completed.returncode})\nstdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}\nerror.json:\n{detail}"
        )
    return json.loads(output_json.read_text(encoding="utf-8"))


def score_for_fold(
    cfg: Config,
    submission_json: str | Path,
    val_image_ids: Sequence[int],
    work_dir: str | Path,
    *,
    tag: str = "fold",
) -> dict:
    """Convenience: build the fold GT, score a submission, return the result."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    gt_json, ignore_json = build_fold_ground_truth(
        cfg, val_image_ids, work_dir / f"{tag}_gt.json", work_dir / f"{tag}_ignore.json"
    )
    return run_official_scorer(
        cfg,
        submission_json=submission_json,
        gt_json=gt_json,
        ignore_json=ignore_json,
        output_json=work_dir / f"{tag}_score.json",
        error_json=work_dir / f"{tag}_error.json",
    )


def final_score(score_payload: dict) -> float:
    """Extract Mask mAP@[0.50:0.95] from an official ``score.json`` payload."""
    evaluation = score_payload.get("evaluation") or {}
    if "score" not in evaluation:
        raise EvalError(
            f"score.json has no evaluation.score; keys={sorted(score_payload)}"
        )
    return float(evaluation["score"])


def summarize(score_payload: dict) -> dict:
    """Compact summary for the console and the experiment log."""
    evaluation = score_payload.get("evaluation") or {}
    validation = score_payload.get("validation") or {}
    per_category = evaluation.get("per_category") or evaluation.get("categories") or {}
    return {
        "mask_map": float(evaluation.get("score", float("nan"))),
        "mask_map_50": evaluation.get("score_50", evaluation.get("map_50")),
        "mask_map_75": evaluation.get("score_75", evaluation.get("map_75")),
        "num_images": validation.get("num_images"),
        "num_predictions": validation.get("num_predictions"),
        "max_predictions_in_one_image": validation.get("max_predictions_in_one_image"),
        "per_category": per_category,
    }


# --------------------------------------------------------------------------- #
# fast local proxy (iteration only - NOT ignore-aware)
# --------------------------------------------------------------------------- #
def fast_local_mask_map(
    cfg: Config,
    submission_json: str | Path,
    val_image_ids: Sequence[int],
    work_dir: str | Path,
    *,
    tag: str = "fast",
) -> float:
    """Plain pycocotools Mask mAP over a fold - quick ranking only.

    Differences from the official number: ignore regions are not applied, and
    maxDets/area defaults follow pycocotools. Use it to compare experiments,
    never to report a score.
    """
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    gt_json, _ = build_fold_ground_truth(
        cfg, val_image_ids, work_dir / f"{tag}_gt.json", work_dir / f"{tag}_ignore.json"
    )

    payload = json.loads(Path(submission_json).read_text(encoding="utf-8"))
    val_names = {image["file_name"] for image in json.loads(
        Path(gt_json).read_text(encoding="utf-8"))["images"]}
    detections = []
    for record in payload.get("results", []):
        if record["image_id"] not in val_names:
            continue
        for instance in record.get("instances", []):
            detections.append(
                {
                    "image_id": record["image_id"],
                    "category_id": int(instance["category_id"]) + 1,  # back to GT numbering
                    "segmentation": instance["segmentation"],
                    "score": float(instance["score"]),
                }
            )

    coco_gt = COCO(str(gt_json))
    # GT uses ids 1/2; map file names to the ids the scorer assigned.
    name_to_id = {image["file_name"]: image["id"] for image in coco_gt.dataset["images"]}
    for detection in detections:
        detection["image_id"] = name_to_id[detection["image_id"]]

    if not detections:
        return 0.0
    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(np.mean(evaluator.eval["precision"][:, :, :, 0, 2]))
