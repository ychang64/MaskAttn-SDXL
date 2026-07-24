#!/usr/bin/env python3
"""Build/cache the paper's multi-noun-phrase COCO caption subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="COCO captions_train2014.json")
    parser.add_argument("--images-dir", required=True, help="COCO train2014 image directory")
    parser.add_argument("--output", default="data/coco_train2014_multi_noun.jsonl")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--min-noun-phrases", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Development-only maximum number of retained examples")
    args = parser.parse_args()
    from maskattn_sdxl.data import build_coco_caption_cache

    retained = build_coco_caption_cache(
        args.annotations,
        args.images_dir,
        args.output,
        min_noun_phrases=args.min_noun_phrases,
        spacy_model=args.spacy_model,
        limit=args.limit,
    )
    print(f"Wrote {retained} retained caption-image pairs to {args.output}")


if __name__ == "__main__":
    main()
