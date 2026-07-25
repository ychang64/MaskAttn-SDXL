#!/usr/bin/env python3
"""Run a bounded MaskAttn-SDXL U-Net forward/backward and pipeline image."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model", default="models/stable-diffusion-xl-base-1.0", help="Local SDXL Diffusers directory")
    value.add_argument("--output-dir", default="outputs/smoke_maskattn_sdxl", help="Ignored-by-Git output directory")
    value.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    value.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Weight dtype; auto selects FP16 on CUDA/MPS and FP32 on CPU.",
    )
    value.add_argument("--height", type=int, default=64, help="Positive multiple of 8")
    value.add_argument("--width", type=int, default=64, help="Positive multiple of 8")
    value.add_argument("--steps", type=int, default=1, help="Denoising steps")
    value.add_argument("--backward", action="store_true", help="Backpropagate through the gate-only MaskAttn U-Net forward")
    value.add_argument("--checkpoint", help="Optional MaskAttn gate checkpoint matching the default smoke configuration")
    value.add_argument("--skip-pipeline", action="store_true", help="Only run the MaskAttn U-Net integration check")
    value.add_argument("--dry-run", action="store_true", help="Validate model layout and print the execution plan")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 8 or args.width % 8:
        raise ValueError("--height and --width must be positive multiples of 8")

    from maskattn_sdxl.runtime import require_local_model, resolve_device, resolve_dtype

    model_path = require_local_model(args.model)
    plan = {
        "model": str(model_path),
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "backward": args.backward,
        "checkpoint": args.checkpoint,
        "pipeline": not args.skip_pipeline,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    import torch
    from diffusers import UNet2DConditionModel

    from maskattn_sdxl.model import (
        MaskAttnConfig,
        assert_only_maskattn_trainable,
        install_maskattn,
        load_maskattn_checkpoint,
    )
    from maskattn_sdxl.runtime import load_sdxl_pipeline

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unet = UNet2DConditionModel.from_pretrained(
        str(model_path), subfolder="unet", variant="fp16", use_safetensors=True, local_files_only=True, torch_dtype=dtype
    ).to(device)
    installation = install_maskattn(unet, MaskAttnConfig())
    if args.checkpoint:
        load_maskattn_checkpoint(unet, args.checkpoint)
    assert_only_maskattn_trainable(unet)
    sample = torch.randn(1, 4, args.height // 8, args.width // 8, device=device, dtype=dtype)
    encoder = torch.randn(1, 77, int(unet.config.cross_attention_dim), device=device, dtype=dtype)
    text_embeds = torch.randn(1, 1280, device=device, dtype=dtype)
    time_ids = torch.tensor([[args.height, args.width, 0, 0, args.height, args.width]], device=device, dtype=dtype)
    kwargs = {"encoder_hidden_states": encoder, "added_cond_kwargs": {"text_embeds": text_embeds, "time_ids": time_ids}}
    if args.backward:
        output = unet(sample, torch.tensor(1, device=device), **kwargs).sample
        output.square().mean().backward()
        backward_ok = any(parameter.grad is not None for parameter in unet.parameters() if parameter.requires_grad)
        if not backward_ok:
            raise RuntimeError("Gate-only backward produced no gradient")
    else:
        with torch.inference_mode():
            output = unet(sample, torch.tensor(1, device=device), **kwargs).sample
        backward_ok = False
    result = {
        "unet_forward_shape": list(output.shape),
        "maskattn_processors": installation.selected_layers,
        "backward": backward_ok,
        "device": device,
    }
    del output, unet
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    if not args.skip_pipeline:
        pipe = load_sdxl_pipeline(model_path, device=device, dtype=args.dtype, local_files_only=True)
        pipeline_installation = install_maskattn(pipe.unet, MaskAttnConfig())
        if args.checkpoint:
            load_maskattn_checkpoint(pipe.unet, args.checkpoint)
        assert_only_maskattn_trainable(pipe.unet)
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
        image_path = output_dir / "maskattn_sdxl_smoke.png"
        image.save(image_path)
        result["pipeline_image"] = str(image_path.resolve())
        result["pipeline_maskattn_processors"] = pipeline_installation.selected_layers
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
