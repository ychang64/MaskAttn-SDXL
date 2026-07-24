#!/usr/bin/env python3
"""Sequential experiment orchestrator; use --dry-run to review all exact commands first."""

from __future__ import annotations

import argparse
import subprocess
import sys

from _common import ROOT, add_common_arguments, execute_mode, resolve_config

DEFAULTS = {
    "train_config": "configs/train_512.yaml",
    "quality_config": "configs/eval_quality.yaml",
    "compositional_config": "configs/eval_compositional.yaml",
    "efficiency_config": "configs/efficiency.yaml",
    "stage_ablation_config": "configs/train_512.yaml",
    "placement_ablation_config": "configs/train_512.yaml",
    "qualitative_config": "configs/generate_qualitative.yaml",
    "baseline_config": "configs/baselines.yaml",
    "run": ["train", "quality", "compositional", "efficiency", "ablate_stage", "ablate_placement", "qualitative", "baseline"],
    "output_dir": "outputs/run_all",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["train", "quality", "compositional", "efficiency", "ablate_stage", "ablate_placement", "qualitative", "baseline"],
        help="Subset to run",
    )
    args = parser.parse_args()
    config = resolve_config(args, DEFAULTS)
    if args.smoke_test:
        if execute_mode(args, config):
            return
    elif args.dry_run:
        # Unlike ordinary entries, this dry-run previews the exact subprocesses.
        selected = args.only or config["run"]
        preview = {
            "train": ("train_maskattn_sdxl.py", config["train_config"]),
            "quality": ("eval_quality.py", config["quality_config"]),
            "compositional": ("eval_compositional.py", config["compositional_config"]),
            "efficiency": ("benchmark_efficiency.py", config["efficiency_config"]),
            "ablate_stage": ("ablate_gating_stage.py", config["stage_ablation_config"]),
            "ablate_placement": ("ablate_module_placement.py", config["placement_ablation_config"]),
            "qualitative": ("generate_qualitative.py", config["qualitative_config"]),
            "baseline": ("generate_baseline.py", config["baseline_config"]),
        }
        for phase in selected:
            script, config_path = preview[phase]
            print(" ".join([sys.executable, str(ROOT / "experiments" / script), "--config", str(ROOT / config_path)]))
        return
    selected = args.only or config["run"]
    mapping = {
        "train": ("train_maskattn_sdxl.py", config["train_config"]),
        "quality": ("eval_quality.py", config["quality_config"]),
        "compositional": ("eval_compositional.py", config["compositional_config"]),
        "efficiency": ("benchmark_efficiency.py", config["efficiency_config"]),
        "ablate_stage": ("ablate_gating_stage.py", config["stage_ablation_config"]),
        "ablate_placement": ("ablate_module_placement.py", config["placement_ablation_config"]),
        "qualitative": ("generate_qualitative.py", config["qualitative_config"]),
        "baseline": ("generate_baseline.py", config["baseline_config"]),
    }
    for phase in selected:
        script, config_path = mapping[phase]
        command = [sys.executable, str(ROOT / "experiments" / script), "--config", str(ROOT / config_path)]
        print("Running:", " ".join(command))
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
