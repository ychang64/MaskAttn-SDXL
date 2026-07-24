#!/usr/bin/env python3
"""Generate fixed multi-object prompts and token-wise MaskAttn gate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT, add_common_arguments, execute_mode, resolve_config

DEFAULTS = {
    "model": {"path": "models/stable-diffusion-xl-base-1.0", "device": "auto", "dtype": "auto"},
    "method": "maskattn",
    "checkpoint": None,
    "maskattn": {"stage": "mid", "placement": "encoder_decoder", "threshold": 0.5, "negative_value": -10000.0, "gate_hidden_dim": 128},
    "prompts": "configs/prompts_qualitative.jsonl",
    "output_dir": "outputs/qualitative",
    "seed": 2026,
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "save_masks": True,
    "comparison": {
        "enabled": True,
        "baseline_method": "sdxl",
        "baseline_model_path": None,
        "baseline_images_dir": None,
    },
}


def _metadata_images(run_dir: Path) -> list[Path]:
    with (run_dir / "metadata.jsonl").open("r", encoding="utf-8") as handle:
        return [Path(json.loads(line)["image"]) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--prompts", help="JSONL prompt file; overrides config.prompts")
    args = parser.parse_args()
    config = resolve_config(args, DEFAULTS)
    if args.prompts:
        config["prompts"] = args.prompts
    if execute_mode(args, config):
        return
    from maskattn_sdxl.generation import generate_records, make_comparison_grid, read_prompt_records

    records = read_prompt_records(config["prompts"])
    run_dir = generate_records(config, records, root=ROOT)
    comparison = config.get("comparison", {})
    if config.get("method") == "maskattn" and comparison.get("enabled", False):
        existing = comparison.get("baseline_images_dir")
        if existing:
            baseline_images = sorted(Path(existing).glob("*.png"))
            if len(baseline_images) != len(records):
                raise ValueError("comparison.baseline_images_dir must contain one PNG per qualitative prompt")
        else:
            baseline_config = dict(config)
            baseline_config["method"] = comparison.get("baseline_method", "sdxl")
            baseline_config["save_masks"] = False
            baseline_config["output_dir"] = str(Path(config["output_dir"]) / "baseline")
            if comparison.get("baseline_model_path"):
                baseline_config["model"] = dict(config["model"])
                baseline_config["model"]["path"] = comparison["baseline_model_path"]
            baseline_dir = generate_records(baseline_config, records, root=ROOT)
            baseline_images = _metadata_images(baseline_dir)
        maskattn_images = _metadata_images(run_dir)
        grid_dir = run_dir / "comparison_grids"
        for index, (baseline_image, maskattn_image) in enumerate(zip(baseline_images, maskattn_images)):
            make_comparison_grid(
                [baseline_image, maskattn_image],
                [comparison.get("baseline_method", "sdxl"), "maskattn_sdxl"],
                grid_dir / f"{index:05d}.png",
            )
    print(run_dir)


if __name__ == "__main__":
    main()
