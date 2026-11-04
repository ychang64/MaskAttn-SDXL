#!/usr/bin/env python3
"""Generate a configured baseline into the same images/metadata.jsonl format as MaskAttn-SDXL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _common import ROOT, add_common_arguments, execute_mode, resolve_config

DEFAULTS = {
    "baseline": "sdxl",
    "baselines": {},
    "device": "auto",
    "dtype": "auto",
    "prompts": "configs/prompts_qualitative.jsonl",
    "output_dir": None,
    "num_samples": None,
    "seed": 2026,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
}


def resolve_baseline(config: dict[str, Any]) -> dict[str, Any]:
    """Merge the selected entry of `baselines` into the config, so a baseline name selects a model."""
    name = config["baseline"]
    entries = config.get("baselines") or {}
    if name not in entries:
        raise ValueError(f"Unknown baseline {name!r}; choose one of: {', '.join(sorted(entries))}")
    entry = entries[name]
    if entry.get("adapter") == "external":
        raise ValueError(f"{name} requires its official repository adapter; see docs/BENCHMARKS.md.")
    return {
        **config,
        "model_path": entry["model_path"],
        "resolution": int(entry["resolution"]),
        "num_inference_steps": entry.get("num_inference_steps", config["num_inference_steps"]),
        "guidance_scale": entry.get("guidance_scale", config["guidance_scale"]),
        "output_dir": config.get("output_dir") or f"outputs/baseline_{name}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(config=ROOT / "configs" / "baselines.yaml")
    parser.add_argument("--baseline", help="Key of the `baselines` entry to generate")
    args = parser.parse_args()
    config = resolve_config(args, DEFAULTS)
    if args.baseline:
        config["baseline"] = args.baseline
    config = resolve_baseline(config)
    if execute_mode(args, config):
        return

    # Runtime-only imports preserve --help/dry-run behavior on systems without Torch.
    import torch
    from diffusers import DiffusionPipeline

    from maskattn_sdxl.config import prepare_run_directory
    from maskattn_sdxl.generation import read_prompt_records
    from maskattn_sdxl.runtime import resolve_device, resolve_dtype, weights_variant

    model_path = Path(config["model_path"])
    if not (model_path / "model_index.json").is_file():
        raise FileNotFoundError(
            f"Baseline model is absent: {model_path}. Run `python scripts/prepare_baselines.py --name {config['baseline']}` "
            "to print its explicit download command."
        )
    device = resolve_device(config["device"])
    torch_dtype = resolve_dtype(config["dtype"], device)
    pipeline = DiffusionPipeline.from_pretrained(
        str(model_path), torch_dtype=torch_dtype, variant=weights_variant(model_path), local_files_only=True
    )
    pipeline.to(device, torch_dtype)
    run_dir = prepare_run_directory(config["output_dir"], config, ROOT)
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    limit = int(config["num_samples"]) if config.get("num_samples") else None
    with (run_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for index, record in enumerate(read_prompt_records(config["prompts"], limit=limit)):
            seed = int(record.get("seed", int(config["seed"]) + index))
            generator = torch.Generator(device=device).manual_seed(seed)
            image = pipeline(
                record["prompt"],
                height=config["resolution"],
                width=config["resolution"],
                num_inference_steps=int(config["num_inference_steps"]),
                guidance_scale=float(config["guidance_scale"]),
                generator=generator,
            ).images[0]
            filename = f"{index:05d}.png"
            image.save(image_dir / filename)
            handle.write(json.dumps({"index": index, "image": str((image_dir / filename).resolve()), "prompt": record["prompt"], "seed": seed, "method": config["baseline"]}) + "\n")
    print(run_dir)


if __name__ == "__main__":
    main()
