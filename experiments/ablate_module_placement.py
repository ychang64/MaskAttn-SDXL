#!/usr/bin/env python3
"""Train encoder-only, decoder-only, and encoder+decoder MaskAttn variants using one model implementation."""

from __future__ import annotations

import argparse
from copy import deepcopy

from _common import ROOT, add_common_arguments, execute_mode, resolve_config
from train_maskattn_sdxl import TRAIN_DEFAULTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--placements", nargs="+", default=["encoder", "decoder", "encoder_decoder"])
    args = parser.parse_args()
    config = resolve_config(args, TRAIN_DEFAULTS)
    if execute_mode(args, config):
        return
    from maskattn_sdxl.training import run_training

    for placement in args.placements:
        if placement not in {"encoder", "decoder", "encoder_decoder"}:
            raise ValueError(f"Unknown placement: {placement}")
        variant = deepcopy(config)
        variant["maskattn"]["placement"] = placement
        variant["output_dir"] = str(config["output_dir"]) + f"_placement_{placement}"
        print(run_training(variant, root=ROOT))


if __name__ == "__main__":
    main()
