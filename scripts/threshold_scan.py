#!/usr/bin/env python
"""Scan ``score_threshold`` on existing fold predictions (E6).

Takes an already-scored submission ``result.json`` (per-instance scores are
threshold-independent), re-filters it at each candidate threshold, and re-scores
with the frozen official scorer. No model inference happens - pure CPU, safe to
run while training occupies the accelerator.

Only meaningful when the input was produced with a *low* threshold (0.05), so
higher cuts are a strict subset. The input predictions must match the GT fold.

Usage
-----
    python scripts/threshold_scan.py \
        --submission outputs/reports/fold0/result.json \
        --gt outputs/reports/fold0/gt.json \
        --ignore outputs/reports/fold0/ignore.json \
        --thresholds 0.05,0.10,0.15,0.20,0.30,0.40,0.50 \
        --out outputs/reports/fold0_threshold_scan.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.config import load_config  # noqa: E402
from eval.mask_map import final_score, run_official_scorer, summarize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True,
                        help="result.json with per-instance scores")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--ignore", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.30,0.40,0.50")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config()
    payload = json.loads(args.submission.read_text(encoding="utf-8"))
    total = sum(len(record.get("instances", [])) for record in payload["results"])

    thresholds = [float(v) for v in args.thresholds.split(",")]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for cut in thresholds:
            filtered = {"version": payload.get("version", "1.0"), "results": []}
            kept = 0
            for record in payload["results"]:
                instances = [i for i in record.get("instances", [])
                             if float(i["score"]) >= cut]
                kept += len(instances)
                filtered["results"].append(
                    {"image_id": record["image_id"], "instances": instances}
                )
            submission = Path(tmp) / f"cut_{cut:.2f}.json"
            submission.write_text(
                json.dumps(filtered, ensure_ascii=False), encoding="utf-8"
            )
            score_payload = run_official_scorer(
                cfg,
                submission_json=submission,
                gt_json=args.gt,
                ignore_json=args.ignore,
                output_json=Path(tmp) / "score.json",
            )
            summary = summarize(score_payload)
            rows.append({
                "threshold": cut,
                "mask_map": final_score(score_payload),
                "num_predictions": summary["num_predictions"],
            })
            print(f"  >= {cut:.2f}  mAP={rows[-1]['mask_map']:.6f}  "
                  f"({kept}/{total} instances kept)")

    best = max(rows, key=lambda row: row["mask_map"])
    report = {"input": str(args.submission), "input_instances": total, "rows": rows,
              "best": best}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\nbest: >= {best['threshold']:.2f}  mAP={best['mask_map']:.6f}")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
