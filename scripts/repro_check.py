#!/usr/bin/env python
"""Offline reproduction spot-check for a packed solution.zip (night assurance).

Extracts the packed archive into a clean directory, builds a small symlinked
image subset, runs the archive's own ``inference.py`` with all fold weights on
CPU, and compares the reproduction against the predictions actually submitted.
Category ids must match exactly; masks are compared by IoU and scores within a
tolerance, because the submitted predictions were produced on a different device
and both shift slightly there. Anything else fails with exit code 1 - the
submission should not be uploaded without a passing check.

Usage
-----
    python scripts/repro_check.py \
        --solution-zip outputs/submissions/solution.zip \
        --predictions outputs/predictions/raw_predictions_testB_ensemble.json \
        --manifest <config: data.test_manifest> \
        --num-images 8
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Mask mAP depends on the segmentation bytes and on the *ranking* of scores, not
# on their absolute value. Re-running the same weights on a different device
# (MPS here, CUDA/CPU for the post-competition review) shifts both: scores move
# in the 1e-6 range, and a handful of boundary pixels flip, which is enough to
# change the RLE bytes of an instance whose mask is otherwise the same.
#
# Demanding byte equality therefore rejects packages that are in fact
# reproducible. Measured on this submission: package-on-CPU is byte-identical to
# the pipeline-on-CPU (min mask IoU 1.000000, score delta 0.0), while either
# against the MPS-produced submission shows 1 of 59 instances at IoU 0.9985.
#
# So: category ids stay strict, scores get a tolerance, and masks are compared by
# IoU - which bounds the metric impact directly, the way the official review
# bounds it (Mask mAP within 0.005).
SCORE_TOLERANCE = 1e-5
MASK_IOU_TOLERANCE = 0.995


def _decode_mask(rle: dict):
    """Decode a compressed-RLE dict to a boolean mask."""
    import numpy as np
    from pycocotools import mask as mask_utils

    packed = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
    return mask_utils.decode(packed).astype(bool)


def _mask_iou(left: dict, right: dict) -> float:
    import numpy as np

    if left == right:
        return 1.0
    a, b = _decode_mask(left), _decode_mask(right)
    union = np.logical_or(a, b).sum()
    if not union:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def compare_instances(expected: list, reproduced: list,
                      score_tolerance: float = SCORE_TOLERANCE,
                      iou_tolerance: float = MASK_IOU_TOLERANCE):
    """Compare one image's instances; return (ok, reason, max score delta, min IoU)."""
    if len(expected) != len(reproduced):
        return False, f"instance count {len(expected)} != {len(reproduced)}", 0.0, 1.0
    max_delta = 0.0
    min_iou = 1.0
    for index, (want, got) in enumerate(zip(expected, reproduced)):
        if want.get("category_id") != got.get("category_id"):
            return (False,
                    f"instance {index}: category_id "
                    f"{want.get('category_id')} != {got.get('category_id')}",
                    max_delta, min_iou)
        iou = _mask_iou(want.get("segmentation"), got.get("segmentation"))
        min_iou = min(min_iou, iou)
        if iou < iou_tolerance:
            return (False,
                    f"instance {index}: mask IoU {iou:.6f} < {iou_tolerance:g}",
                    max_delta, min_iou)
        delta = abs(float(want.get("score", 0.0)) - float(got.get("score", 0.0)))
        max_delta = max(max_delta, delta)
        if delta > score_tolerance:
            return (False,
                    f"instance {index}: score {want.get('score')} != "
                    f"{got.get('score')} (delta {delta:.3g} > {score_tolerance:g})",
                    max_delta, min_iou)
    return True, "", max_delta, min_iou


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-zip", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    for required in (args.solution_zip, args.predictions, args.manifest):
        if not required.is_file():
            raise SystemExit(f"input missing: {required}")

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))[: args.num_images]
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))

    workdir = Path(tempfile.mkdtemp(prefix="night_repro_"))
    pkg_dir = workdir / "pkg"
    img_dir = workdir / "imgs"
    pkg_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    try:
        subprocess.run(
            ["unzip", "-q", str(args.solution_zip), "-d", str(pkg_dir)], check=True
        )
        weights = sorted(
            (pkg_dir / "model").glob("fold*.pt"),
            key=lambda p: int(p.stem[len("fold"):]),
        )
        if not weights:
            raise SystemExit(f"no fold*.pt weights inside {args.solution_zip}")

        for row in rows:
            (img_dir / row["image_id"]).symlink_to(
                (REPO_ROOT / "data" / "testB" / "testB" / "images"
                 / row["image_id"]).resolve()
            )

        manifest_subset = workdir / "manifest.json"
        manifest_subset.write_text(json.dumps(rows), encoding="utf-8")
        repro_path = workdir / "repro.json"
        subprocess.run(
            [
                sys.executable, str(pkg_dir / "inference.py"),
                "--input_dir", str(img_dir),
                "--weights", *[str(w) for w in weights],
                "--output_path", str(repro_path),
                "--manifest", str(manifest_subset),
                "--device", args.device,
            ],
            check=True,
            cwd=str(pkg_dir),
        )

        repro = json.loads(repro_path.read_text(encoding="utf-8"))
        repro_map = {r["image_id"]: r["instances"] for r in repro["results"]}

        report = {
            "checked_images": len(rows),
            "score_tolerance": SCORE_TOLERANCE,
            "mask_iou_tolerance": MASK_IOU_TOLERANCE,
            "max_score_delta": 0.0,
            "min_mask_iou": 1.0,
            "mismatches": [],
        }
        for row in rows:
            image_id = row["image_id"]
            expected = predictions.get(image_id, [])
            actual = repro_map.get(image_id)
            if actual is None:
                report["mismatches"].append({
                    "image_id": image_id, "reason": "image missing from reproduction",
                })
                continue
            ok, reason, delta, iou = compare_instances(expected, actual)
            report["max_score_delta"] = max(report["max_score_delta"], delta)
            report["min_mask_iou"] = min(report["min_mask_iou"], iou)
            if not ok:
                report["mismatches"].append({
                    "image_id": image_id,
                    "expected_instances": len(expected),
                    "reproduced_instances": len(actual),
                    "reason": reason,
                })
        report["ok"] = not report["mismatches"]
        report_path = workdir / "repro_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        shutil.copy(report_path, REPO_ROOT / "outputs" / "reports" / "repro_check.json")

        if report["ok"]:
            print(f"repro check PASS: {len(rows)} images reproduced "
                  f"({len(weights)} weights, device={args.device}); "
                  f"categories identical, min mask IoU "
                  f"{report['min_mask_iou']:.6f} >= {MASK_IOU_TOLERANCE:g}, "
                  f"max score delta {report['max_score_delta']:.3g} <= "
                  f"{SCORE_TOLERANCE:g}")
            return 0
        print(f"repro check FAIL: {report['mismatches']}")
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
