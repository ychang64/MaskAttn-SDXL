#!/usr/bin/env python3
"""Generate a configured baseline into the same images/metadata.jsonl format as MaskAttn-SDXL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT, add_common_arguments, execute_mode, resolve_config

DEFAULTS = {
    "baseline": "sdxl",
    "model_path": "models/stable-diffusion-xl-base-1.0",
    "device": "auto",
    "dtype": "auto",
    "prompts": "configs/prompts_qualitative.jsonl",
    "output_dir": "outputs/baseline_sdxl",
    "seed": 2026,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--baseline", help="Baseline name recorded in metadata")
    args = parser.parse_args()
    config = resolve_config(args, DEFAULTS)
    if args.baseline:
        config["baseline"] = args.baseline
    if execute_mode(args, config):
        return

    # Runtime-only imports preserve --help/dry-run behavior on systems without Torch.
    import torch
    from diffusers import DiffusionPipeline

    from maskattn_sdxl.config import prepare_run_directory
    from maskattn_sdxl.generation import read_prompt_records
    from maskattn_sdxl.runtime import resolve_device, resolve_dtype

    model_path = Path(config["model_path"])
    if not (model_path / "model_index.json").is_file():
        raise FileNotFoundError(
            f"Baseline model is absent: {model_path}. Run `python scripts/prepare_baselines.py --name {config['baseline']}` "
            "to print its explicit download command."
        )
    device = resolve_device(config["device"])
    pipeline = DiffusionPipeline.from_pretrained(str(model_path), torch_dtype=resolve_dtype(config["dtype"], device), local_files_only=True)
    pipeline.to(device)
    run_dir = prepare_run_directory(config["output_dir"], config, ROOT)
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for index, record in enumerate(read_prompt_records(config["prompts"])):
            seed = int(record.get("seed", int(config["seed"]) + index))
            generator = torch.Generator(device=device).manual_seed(seed)
            image = pipeline(
                record["prompt"],
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
