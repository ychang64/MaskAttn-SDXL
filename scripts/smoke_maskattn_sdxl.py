#!/usr/bin/env python3
"""Run a bounded MaskAttn-SDXL U-Net forward/backward and pipeline image."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model", default="models/stable-diffusion-xl-base-1.0", help="Local SDXL Diffusers directory")
    value.add_argument("--output-dir", default="outputs/smoke_maskattn_sdxl", help="Ignored-by-Git output directory")
    value.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    value.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    value.add_argument("--height", type=int, default=64, help="Positive multiple of 8")
    value.add_argument("--width", type=int, default=64, help="Positive multiple of 8")
    value.add_argument("--steps", type=int, default=1, help="Denoising steps")
    value.add_argument("--backward", action="store_true", help="Backpropagate through the gate-only MaskAttn U-Net forward")
    value.add_argument("--checkpoint", help="Trained MaskAttn gate checkpoint")
    value.add_argument(
        "--allow-untrained-gates",
        action="store_true",
        help="Explicitly permit random gates for integration wiring only",
    )
    value.add_argument(
        "--allow-test-only-checkpoint",
        action="store_true",
        help="Permit a checkpoint marked test_only for loader verification; never use this for evaluation",
    )
    value.add_argument("--skip-pipeline", action="store_true", help="Only run the final MaskAttn U-Net integration check")
    value.add_argument("--dry-run", action="store_true", help="Validate model layout and print the execution plan")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 8 or args.width % 8:
        raise ValueError("--height and --width must be positive multiples of 8")
    from maskattn_sdxl.runtime import require_local_model

    model_path = require_local_model(args.model)
    plan = {
        "model": str(model_path),
        "device": args.device,
        "dtype": args.dtype,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "backward": args.backward,
        "checkpoint": args.checkpoint,
        "allow_untrained_gates": args.allow_untrained_gates,
        "allow_test_only_checkpoint": args.allow_test_only_checkpoint,
        "pipeline": not args.skip_pipeline,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if args.checkpoint is None and not args.allow_untrained_gates:
        raise ValueError("Provide --checkpoint or explicitly pass --allow-untrained-gates for integration-only smoke")

    import torch

    from maskattn_sdxl.model import MaskAttnConfig, assert_maskattn_ready
    from maskattn_sdxl.runtime import load_maskattn_pipeline, resolve_device, resolve_dtype

    device = resolve_device(args.device)

    pipe, _, runtime_audit = load_maskattn_pipeline(
        model_path,
        checkpoint=args.checkpoint,
        maskattn_config=None if args.checkpoint else asdict(MaskAttnConfig()),
        allow_untrained_gates=args.allow_untrained_gates,
        allow_test_only_checkpoint=args.allow_test_only_checkpoint,
        device=device,
        dtype=args.dtype,
        local_files_only=True,
    )
    runtime_method = runtime_audit["method"]
    unet = pipe.unet
    dtype = resolve_dtype(args.dtype, device)
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
    runtime_audit = assert_maskattn_ready(
        unet,
        checkpoint_required=runtime_audit["trained_checkpoint_loaded"],
        require_forward_calls=True,
    )
    result = {
        **plan,
        "method": runtime_method,
        "final_unet_id": id(unet),
        "unet_forward_shape": list(output.shape),
        "backward": backward_ok,
        "runtime_audit": runtime_audit,
    }
    if not args.skip_pipeline:
        with torch.inference_mode():
            image = pipe(
                "A red square on the left and a blue circle on the right",
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=1.0,
                generator=torch.Generator(device=device).manual_seed(17),
            ).images[0]
        runtime_audit = assert_maskattn_ready(
            unet,
            checkpoint_required=runtime_audit["trained_checkpoint_loaded"],
            require_forward_calls=True,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "maskattn_sdxl_smoke.png"
        image.save(image_path)
        result["pipeline_image"] = str(image_path.resolve())
        result["runtime_audit"] = runtime_audit
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
