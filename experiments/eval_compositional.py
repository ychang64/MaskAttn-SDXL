#!/usr/bin/env python3
"""Run an official T2I-CompBench++ Spatial/UniDet or GenEval evaluator through a configured adapter."""

from __future__ import annotations

import argparse

from _common import add_common_arguments, execute_mode, resolve_config

DEFAULTS = {
    "benchmark": "t2i_compbench_spatial",
    "images_dir": "outputs/eval_compositional/images",
    "prompts": "data/t2i_compbench_spatial_prompts.jsonl",
    "output_dir": "outputs/eval_compositional/metrics",
    "evaluator_command": None,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--evaluator-command", help="Official evaluator command with {images}, {prompts}, {output}")
    args = parser.parse_args()
    config = resolve_config(args, DEFAULTS)
    if args.evaluator_command:
        config["evaluator_command"] = args.evaluator_command
    if execute_mode(args, config):
        return
    from maskattn_sdxl.benchmarks import run_external_evaluator

    output = run_external_evaluator(
        config.get("evaluator_command"),
        images_dir=config["images_dir"],
        prompts_path=config["prompts"],
        output_dir=config["output_dir"],
        benchmark_name=config["benchmark"],
    )
    print(output)


if __name__ == "__main__":
    main()
