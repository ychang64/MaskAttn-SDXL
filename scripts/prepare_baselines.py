#!/usr/bin/env python3
"""Print (or explicitly execute) the model preparation command for a named baseline."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs" / "baselines.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--execute", action="store_true", help="Actually run the displayed command")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        baselines = yaml.safe_load(handle)["baselines"]
    if args.name not in baselines:
        raise ValueError(f"Unknown baseline {args.name}; choose one of: {', '.join(sorted(baselines))}")
    entry = baselines[args.name]
    command = entry.get("prepare_command")
    if not command:
        raise ValueError(
            f"{args.name} requires its official repository adapter; no generic checkpoint download is declared. See docs/BENCHMARKS.md."
        )
    print(command)
    if args.execute:
        raise SystemExit(subprocess.run(shlex.split(command), cwd=ROOT, check=False).returncode)


if __name__ == "__main__":
    main()
