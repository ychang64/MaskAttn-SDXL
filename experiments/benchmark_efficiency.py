#!/usr/bin/env python3
"""Measure matched-sampling parameter count, CUDA peak memory, and latency for SDXL and MaskAttn-SDXL."""

from __future__ import annotations

import argparse
import json

from _common import add_common_arguments, execute_mode, resolve_config

DEFAULTS = {
    "model": {"path": "models/stable-diffusion-xl-base-1.0", "device": "auto", "dtype": "auto"},
    "maskattn": {"stage": "mid", "placement": "encoder_decoder", "threshold": 0.5, "negative_value": -10000.0, "gate_hidden_dim": 128},
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
    from maskattn_sdxl.benchmarks import benchmark_pipeline, write_json
    from maskattn_sdxl.generation import _load_pipeline_for_generation

    base_config = dict(config)
    base_config["method"] = "sdxl"
    base_pipe, _, _ = _load_pipeline_for_generation(base_config)
    mask_config = dict(config)
    mask_config["method"] = "maskattn"
    mask_pipe, _, _ = _load_pipeline_for_generation(mask_config)
    common = {"prompt": config["prompt"], "warmup": int(config["warmup"]), "repeats": int(config["repeats"]), "steps": int(config["num_inference_steps"]), "guidance_scale": float(config["guidance_scale"])}
    results = {"sdxl": benchmark_pipeline(base_pipe, **common), "maskattn_sdxl": benchmark_pipeline(mask_pipe, **common)}
    output = write_json(results, config["output_dir"] + "/efficiency.json")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
