#!/usr/bin/env python3
"""Fine-tune MaskAttn gate heads while keeping SDXL frozen."""

from __future__ import annotations

import argparse

from _common import ROOT, add_common_arguments, execute_mode, resolve_config

TRAIN_DEFAULTS = {
    "model": {"path": "models/stable-diffusion-xl-base-1.0", "device": "auto", "dtype": "auto", "allow_download": False},
    "data": {"caption_cache": "data/coco_train2014_multi_noun.jsonl", "num_workers": 4},
    "maskattn": {"stage": "mid", "placement": "encoder_decoder", "threshold": 0.5, "negative_value": -10000.0, "gate_hidden_dim": 128},
    "train": {"resolution": 512, "max_train_steps": 100000, "train_batch_size": 2, "gradient_accumulation_steps": 8, "learning_rate": 0.0001, "weight_decay": 0.01, "betas": [0.9, 0.999], "lr_warmup_steps": 1000, "mixed_precision": "fp16", "max_grad_norm": 1.0, "seed": 42, "checkpointing_steps": 1000, "log_every": 20, "validation_every": 1000, "validation_prompts": ["A red dragon on the left and a blue dragon on the right, cinematic shot"], "resume_from": None},
    "output_dir": "outputs/train_maskattn_sdxl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    config = resolve_config(args, TRAIN_DEFAULTS)
    if execute_mode(args, config):
        return
    from maskattn_sdxl.training import run_training

    checkpoint = run_training(config, root=ROOT)
    print(checkpoint)


if __name__ == "__main__":
    main()
