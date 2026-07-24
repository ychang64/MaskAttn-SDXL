"""Adapters for official composition evaluators and reproducible efficiency measurements."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


def run_external_evaluator(
    command_template: str,
    *,
    images_dir: str | Path,
    prompts_path: str | Path,
    output_dir: str | Path,
    benchmark_name: str,
) -> Path:
    """Call an official benchmark evaluator supplied by the user/configuration.

    No substitute score is computed. The command receives ``{images}``, ``{prompts}``, and ``{output}`` placeholders.
    """
    if not command_template:
        raise ValueError(
            f"{benchmark_name} evaluator is not configured. Install the official evaluator and set `evaluator_command`; "
            "see docs/BENCHMARKS.md for required placeholders."
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    command = command_template.format(images=Path(images_dir).resolve(), prompts=Path(prompts_path).resolve(), output=output.resolve())
    result = subprocess.run(shlex.split(command), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{benchmark_name} evaluator failed with exit code {result.returncode}: {command}")
    return output


def benchmark_pipeline(pipe, *, prompt: str, warmup: int, repeats: int, steps: int, guidance_scale: float) -> dict[str, Any]:
    import torch

    device = pipe._execution_device
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    for _ in range(warmup):
        pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    result: dict[str, Any] = {"latency_seconds_per_image": elapsed / repeats, "warmup": warmup, "repeats": repeats, "steps": steps}
    if str(device).startswith("cuda"):
        result["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    else:
        result["peak_gpu_memory_bytes"] = None
        result["memory_note"] = "Peak GPU memory is only available for CUDA runs."
    result["parameter_count"] = int(sum(parameter.numel() for parameter in pipe.unet.parameters()))
    return result


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return target


def write_metrics_csv(payload: dict[str, Any], path: str | Path) -> Path:
    """Write scalar metric aggregates in a portable one-row CSV without inventing values."""
    import csv

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scalars = {key: value for key, value in payload.items() if isinstance(value, (str, int, float)) or value is None}
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalars))
        writer.writeheader()
        writer.writerow(scalars)
    return target
