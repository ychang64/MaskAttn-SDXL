#!/usr/bin/env python3
"""Generate one bounded baseline SDXL image from a local Diffusers checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model", default="models/stable-diffusion-xl-base-1.0", help="Local SDXL Diffusers directory")
    value.add_argument("--output-dir", default="outputs/smoke_sdxl", help="Ignored-by-Git output directory")
    value.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    value.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    value.add_argument("--height", type=int, default=64, help="Positive multiple of 8")
    value.add_argument("--width", type=int, default=64, help="Positive multiple of 8")
    value.add_argument("--steps", type=int, default=1, help="Denoising steps")
    value.add_argument("--dry-run", action="store_true", help="Validate model layout and print the execution plan")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 8 or args.width % 8:
        raise ValueError("--height and --width must be positive multiples of 8")

    from maskattn_sdxl.runtime import load_sdxl_pipeline, require_local_model, resolve_device

    model_path = require_local_model(args.model)
    device = resolve_device(args.device)
    plan = {
        "model": str(model_path),
        "device": device,
        "dtype": args.dtype,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    import torch

    pipe = load_sdxl_pipeline(model_path, device=device, dtype=args.dtype, local_files_only=True)
    pipe.enable_attention_slicing()
    with torch.inference_mode():
        image = pipe(
            "A red square on the left and a blue circle on the right",
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            generator=torch.Generator(device=device).manual_seed(17),
        ).images[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "sdxl_smoke.png"
    image.save(image_path)
    result = {**plan, "pipeline_image": str(image_path.resolve())}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
