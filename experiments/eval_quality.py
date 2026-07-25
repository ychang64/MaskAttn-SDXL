#!/usr/bin/env python3
"""Generate caption-conditioned images then compute FID, CLIP Score, precision, and recall."""

from __future__ import annotations

import argparse
import json

from _common import ROOT, add_common_arguments, execute_mode, resolve_config

DEFAULTS = {
    "model": {"path": "models/stable-diffusion-xl-base-1.0", "device": "auto", "dtype": "auto"},
    "method": "maskattn",
    "checkpoint": None,
    "maskattn": {"stage": "mid", "placement": "encoder_decoder", "threshold": 0.5, "negative_value": -10000.0, "gate_hidden_dim": 128},
    "prompts": "data/coco_val2014_3000_prompts.jsonl",
    "real_images": "data/coco_val2014_images",
    "output_dir": "outputs/eval_quality_coco",
    "num_samples": 3000,
    "seed": 42,
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "save_masks": False,
    "metric_device": "cuda",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--images-dir", help="Reuse an existing generated image directory instead of generating")
    parser.add_argument("--skip-metrics", action="store_true", help="Generate images/metadata only")
    args = parser.parse_args()
    config = resolve_config(args, DEFAULTS)
    if execute_mode(args, config):
        return
    if config.get("method", "maskattn") == "maskattn" and not config.get("checkpoint"):
        raise ValueError("MaskAttn quality evaluation requires a trained `checkpoint`; random gates are not valid evaluation.")
    from maskattn_sdxl.benchmarks import write_json, write_metrics_csv
    from maskattn_sdxl.generation import generate_records, read_prompt_records
    from maskattn_sdxl.metrics import compute_clip_score, compute_fid_and_precision_recall

    records = read_prompt_records(config["prompts"], limit=int(config["num_samples"]))
    if args.images_dir:
        image_dir = args.images_dir
    else:
        run_dir = generate_records(config, records, root=ROOT)
        image_dir = run_dir / "images"
    if args.skip_metrics:
        print(image_dir)
        return
    results = compute_fid_and_precision_recall(image_dir, config["real_images"], device=config.get("metric_device", "cuda"))
    results["clip_score"] = compute_clip_score(image_dir, [record["prompt"] for record in records], device=config.get("metric_device", "cuda"))
    output = write_json(results, config["output_dir"] + "/metrics.json")
    csv_output = write_metrics_csv(results, config["output_dir"] + "/metrics.csv")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(output)
    print(csv_output)


if __name__ == "__main__":
    main()
