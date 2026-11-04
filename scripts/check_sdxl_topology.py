#!/usr/bin/env python3
"""Exercise MaskAttn on the full-size SDXL U-Net with random weights; no checkpoint download is needed.

This checks wiring, mixed precision, gradient flow, and GPU memory. Random weights say nothing about image quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]


def _gib(size: int) -> float:
    return round(size / 1024**3, 2)


def _inputs(batch: int, resolution: int, device: str):
    import torch

    return {
        "sample": torch.randn(batch, 4, resolution // 8, resolution // 8, device=device).half(),
        "encoder_hidden_states": torch.randn(batch, 77, 2048, device=device).half(),
        "added_cond_kwargs": {
            "text_embeds": torch.randn(batch, 1280, device=device).half(),
            "time_ids": torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]], device=device).repeat(batch, 1).half(),
        },
    }


def check_identity(unet, device: str) -> float:
    """Largest output change caused by installing new gates, which must be numerically zero."""
    import torch

    from maskattn_sdxl.model import MaskAttnConfig, install_maskattn

    inputs = _inputs(1, 512, device)
    timestep = torch.tensor(500, device=device)
    with torch.no_grad():
        base = unet(timestep=timestep, **inputs).sample
        install_maskattn(unet, MaskAttnConfig())
        gated = unet(timestep=timestep, **inputs).sample
    return float((base - gated).abs().max())


def check_training(unet, device: str, *, batch: int, resolution: int, steps: int, checkpointing: bool) -> dict:
    import torch

    from maskattn_sdxl.model import gate_heads, maskattn_open_fraction, maskattn_parameters

    if checkpointing:
        unet.enable_gradient_checkpointing()
        unet.train()
    else:
        unet.disable_gradient_checkpointing()
        unet.eval()
    parameters = maskattn_parameters(unet)
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    for _ in range(steps):
        inputs = _inputs(batch, resolution, device)
        noise = torch.randn_like(inputs["sample"].float())
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = unet(timestep=torch.randint(0, 1000, (batch,), device=device), **inputs).sample
            loss = torch.nn.functional.mse_loss(prediction.float(), noise)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    heads = gate_heads(unet)
    return {
        "batch": batch,
        "resolution": resolution,
        "gradient_checkpointing": checkpointing,
        "seconds_per_step": round((time.perf_counter() - started) / steps, 3),
        "peak_gpu_gib": _gib(torch.cuda.max_memory_allocated()),
        "finite_loss": all(torch.isfinite(torch.tensor(value)) for value in losses),
        "last_gradient_norm": float(norm),
        "changed_parameters": sum(not torch.equal(old, new) for old, new in zip(before, parameters)),
        "total_parameter_tensors": len(parameters),
        "gate_heads": len(heads),
        "gate_open_fraction": maskattn_open_fraction(unet),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--runs", nargs="+", default=["2x512", "1x1024:gc"], help="BATCHxRESOLUTION, with a :gc suffix for gradient checkpointing")
    parser.add_argument(
        "--memory-limit-gib",
        type=float,
        help="Fail with out-of-memory above this allocation. On Windows, VRAM overcommit otherwise spills to system RAM and crawls.",
    )
    args = parser.parse_args()

    import torch
    from tiny_sdxl import build_sdxl_unet

    from maskattn_sdxl.model import assert_only_maskattn_trainable, maskattn_parameters

    device = "cuda"
    if args.memory_limit_gib:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_limit_gib * 1024**3 / total))
    torch.manual_seed(0)
    unet = build_sdxl_unet().to(device, torch.float16)
    unet.requires_grad_(False)
    report = {"max_output_change_from_new_gates": check_identity(unet, device)}
    assert_only_maskattn_trainable(unet)
    report["gate_parameters"] = sum(parameter.numel() for parameter in maskattn_parameters(unet))
    report["unet_gib_at_rest"] = _gib(torch.cuda.memory_allocated())
    print(json.dumps(report), flush=True)
    for run in args.runs:
        shape, _, flag = run.partition(":")
        batch, resolution = (int(value) for value in shape.split("x"))
        try:
            result = check_training(unet, device, batch=batch, resolution=resolution, steps=args.steps, checkpointing=flag == "gc")
        except torch.OutOfMemoryError:
            result = {"run": run, "result": "out of memory"}
            torch.cuda.empty_cache()
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
