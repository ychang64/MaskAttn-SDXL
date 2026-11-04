#!/usr/bin/env python3
"""Sample the Table I evaluation captions and copy their matching real images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_COUNTS = {"coco": 3000, "flickr30k": 500}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_COUNTS), required=True)
    parser.add_argument("--captions", required=True, help="COCO captions_val2014.json, or Flickr30k results_20130124.token")
    parser.add_argument("--images-dir", required=True, help="Directory holding the dataset's images")
    parser.add_argument("--output-prompts", required=True, help="Prompt JSONL read by eval_quality.py")
    parser.add_argument("--output-images", required=True, help="Directory for the matching real images")
    parser.add_argument("--count", type=int, help="Number of images; defaults to the paper's split size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from maskattn_sdxl.data import build_coco_eval_split, build_flickr30k_eval_split

    build = build_coco_eval_split if args.dataset == "coco" else build_flickr30k_eval_split
    written = build(
        args.captions,
        args.images_dir,
        args.output_prompts,
        args.output_images,
        count=args.count or DEFAULT_COUNTS[args.dataset],
        seed=args.seed,
    )
    print(f"Wrote {written} prompts to {args.output_prompts} and real images to {args.output_images}")


if __name__ == "__main__":
    main()
