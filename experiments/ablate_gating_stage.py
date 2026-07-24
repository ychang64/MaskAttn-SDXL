#!/usr/bin/env python3
"""Train high/mid/low/all MaskAttn stage variants through one shared implementation."""

from __future__ import annotations

import argparse
from copy import deepcopy

from _common import ROOT, add_common_arguments, execute_mode, resolve_config
from train_maskattn_sdxl import TRAIN_DEFAULTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--stages", nargs="+", default=["high", "mid", "low", "all"])
    args = parser.parse_args()
    config = resolve_config(args, TRAIN_DEFAULTS)
    if execute_mode(args, config):
        return
    from maskattn_sdxl.training import run_training

    for stage in args.stages:
        if stage not in {"high", "mid", "low", "all"}:
            raise ValueError(f"Unknown stage: {stage}")
        variant = deepcopy(config)
        variant["maskattn"]["stage"] = stage
        variant["output_dir"] = str(config["output_dir"]) + f"_stage_{stage}"
        print(run_training(variant, root=ROOT))


if __name__ == "__main__":
    main()
