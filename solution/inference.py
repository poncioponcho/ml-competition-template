#!/usr/bin/env python3
"""Unified inference entry point (official contract).

    python inference.py --input_dir test_images/ --weights model/weights.pt \
        --output_path reproduced_result.json

Reads every image under ``--input_dir`` and writes a ``result.json`` in the
submission format. Every image gets exactly one record; images with no
detection get ``"instances": []``.

Without ``--weights`` it emits format-valid empty predictions, which is the
"model not trained yet" state: the official validator accepts it and the score
is 0. That safety net exists so a late failure can never leave the team with
nothing to submit.

This file ships inside ``solution.zip`` and must reproduce the submitted
predictions **offline**. Keep it self-contained: no network access, no imports
from outside the archive.

To plug in a model, implement ``build_predictor`` (see the ``TODO`` below) and
return a callable ``(image_path, width, height) -> list[instance]``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def load_metadata(manifest_path: Path | None) -> dict[str, tuple[int, int]]:
    """Optional public manifest giving per-image width/height."""
    if manifest_path is None:
        return {}
    rows = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {row["image_id"]: (int(row["width"]), int(row["height"])) for row in rows}


def build_predictor(weights, device: str, short_side=None):
    """Return a callable ``(image_path, width, height) -> list[instance]``.

    Missing weights are a hard error: silently emitting empty predictions when
    weights were requested would fake a reproduction.
    """
    if weights is None:
        def empty_predictor(_path, _width, _height):
            return []
        return empty_predictor

    paths = list(weights) if isinstance(weights, (list, tuple)) else [weights]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"weights not found: {missing}")

    # TODO: replace with your own model.
    #
    #   import torch
    #   from model.model import Segmenter          # packaged alongside this file
    #   model = Segmenter.load(paths, device=device, short_side=short_side)
    #   model.eval()
    #
    #   def predictor(image_path, width, height):
    #       return model.predict(image_path, width, height)
    #
    #   return predictor
    #
    # Two rules that decide whether the post-competition reproduction matches:
    #   * inference settings (resolution, threshold, scales, weight order) must
    #     equal the ones the submission was produced with;
    #   * masks must be resized back to the ORIGINAL image size before RLE
    #     encoding, as the official instructions require.
    raise SystemExit(
        "--weights given but no model is wired up yet: implement build_predictor()"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, type=Path,
                        help="directory of test images")
    parser.add_argument("--weights", nargs="+", default=None, type=Path,
                        help="model weights, one or more (multiple checkpoints "
                             "are ensembled by per-class NMS; omit to emit "
                             "empty predictions)")
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--manifest", default=None, type=Path,
                        help="optional public manifest for exact dimensions")
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--short_side", default=None,
                        help="inference resolution(s), e.g. 1024 or 640,1024; "
                             "defaults to what the submission used")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"input_dir not found: {args.input_dir}")

    metadata = load_metadata(args.manifest)
    images = sorted(
        path for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise SystemExit(f"no images found under {args.input_dir}")

    predict = build_predictor(args.weights, args.device, args.short_side)

    results = []
    total_instances = 0
    for index, image_path in enumerate(images, 1):
        name = image_path.name
        width, height = metadata.get(name, (0, 0))
        if not width or not height:
            from PIL import Image
            with Image.open(image_path) as handle:
                width, height = handle.size
        instances = predict(image_path, width, height)
        instances = [item for item in instances if item["score"] >= args.score_threshold]
        instances.sort(key=lambda item: item["score"], reverse=True)
        total_instances += len(instances)
        results.append({"image_id": name, "instances": instances})
        if index % 20 == 0 or index == len(images):
            print(f"  {index}/{len(images)} images", file=sys.stderr)

    payload = {"version": "1.0", "results": results}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"wrote {args.output_path} ({len(results)} images, {total_instances} instances)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
