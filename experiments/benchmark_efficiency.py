#!/usr/bin/env python3
"""Measure matched-sampling parameter count, CUDA peak memory, and latency for SDXL and MaskAttn-SDXL."""

from __future__ import annotations

import argparse
import gc
import json

from _common import MASKATTN_DEFAULTS, add_common_arguments, execute_mode, resolve_config

DEFAULTS = {
    "model": {"path": "models/stable-diffusion-xl-base-1.0", "device": "auto", "dtype": "auto"},
    "maskattn": dict(MASKATTN_DEFAULTS),
    "checkpoint": None,
    "prompt": "A red dragon on the left and a blue dragon on the right",
    "warmup": 3,
    "repeats": 10,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "output_dir": "outputs/efficiency",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    config = resolve_config(args, DEFAULTS)
    if execute_mode(args, config):
        return
    if not config.get("checkpoint"):
        raise ValueError("MaskAttn efficiency benchmarking requires a trained `checkpoint`; random gates are not benchmarked.")
    import torch

    from maskattn_sdxl.benchmarks import benchmark_pipeline, write_json
    from maskattn_sdxl.generation import _load_pipeline_for_generation

    common = {"prompt": config["prompt"], "warmup": int(config["warmup"]), "repeats": int(config["repeats"]), "steps": int(config["num_inference_steps"]), "guidance_scale": float(config["guidance_scale"])}
    results = {}
    # One pipeline at a time: a second pipeline on the GPU would be counted in the first one's peak memory.
    for label, method in (("sdxl", "sdxl"), ("maskattn_sdxl", "maskattn")):
        pipe, _, _ = _load_pipeline_for_generation({**config, "method": method})
        results[label] = benchmark_pipeline(pipe, **common)
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = write_json(results, config["output_dir"] + "/efficiency.json")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
