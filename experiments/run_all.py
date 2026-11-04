#!/usr/bin/env python3
"""Sequential experiment orchestrator; use --dry-run to review all exact commands first."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any

from _common import ROOT, add_common_arguments, execute_mode, resolve_config

# phase -> (entry script, config key)
PHASES = {
    "train_512": ("train_maskattn_sdxl.py", "train_512_config"),
    "train_1024": ("train_maskattn_sdxl.py", "train_1024_config"),
    "quality": ("eval_quality.py", "quality_config"),
    "compositional": ("eval_compositional.py", "compositional_config"),
    "efficiency": ("benchmark_efficiency.py", "efficiency_config"),
    "ablate_stage": ("ablate_gating_stage.py", "stage_ablation_config"),
    "ablate_placement": ("ablate_module_placement.py", "placement_ablation_config"),
    "qualitative": ("generate_qualitative.py", "qualitative_config"),
    "baseline": ("generate_baseline.py", "baseline_config"),
}
PHASES_USING_GATES = {"quality", "efficiency", "qualitative"}

DEFAULTS = {
    "train_512_config": "configs/train_512.yaml",
    "train_1024_config": "configs/train_1024.yaml",
    "quality_config": "configs/eval_quality.yaml",
    "compositional_config": "configs/eval_compositional.yaml",
    "efficiency_config": "configs/efficiency.yaml",
    "stage_ablation_config": "configs/train_512.yaml",
    "placement_ablation_config": "configs/train_512.yaml",
    "qualitative_config": "configs/generate_qualitative.yaml",
    "baseline_config": "configs/baselines.yaml",
    "checkpoint": "outputs/train_1024/checkpoints/gate_final.pt",
    # Compositional scoring needs an external evaluator, and each ablation variant is a full training run.
    "run": ["train_512", "train_1024", "quality", "efficiency", "qualitative"],
    "output_dir": "outputs/run_all",
}


def build_command(phase: str, config: dict[str, Any]) -> list[str]:
    script, config_key = PHASES[phase]
    command = [sys.executable, str(ROOT / "experiments" / script), "--config", str(ROOT / config[config_key])]
    if phase in PHASES_USING_GATES:
        command += ["--set", f"checkpoint={config['checkpoint']}"]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--only", nargs="+", choices=sorted(PHASES), help="Subset to run")
    args = parser.parse_args()
    config = resolve_config(args, DEFAULTS)
    selected = args.only or config["run"]
    if args.dry_run:
        # Unlike ordinary entries, this dry-run previews the exact subprocesses.
        for phase in selected:
            print(" ".join(build_command(phase, config)))
        return
    if execute_mode(args, config):
        return
    for phase in selected:
        command = build_command(phase, config)
        print("Running:", " ".join(command))
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
