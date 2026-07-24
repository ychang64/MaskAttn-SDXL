"""Shared standard-library CLI setup. Keep this file lightweight so --help needs no ML runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a nested YAML value")
    parser.add_argument("--output-dir", type=Path, help="Override config.output_dir")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print resolved configuration without ML imports")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny CPU gate forward/backward test; no data/model required")


def resolve_config(args: argparse.Namespace, defaults: dict[str, Any]) -> dict[str, Any]:
    from maskattn_sdxl.config import deep_update, load_yaml, parse_overrides

    config = deep_update(defaults, load_yaml(args.config))
    config = deep_update(config, parse_overrides(args.set))
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    return config


def execute_mode(args: argparse.Namespace, config: dict[str, Any]) -> bool:
    """Return true when a lightweight mode completed and the caller should exit."""
    if args.dry_run:
        import json

        print(json.dumps(config, indent=2, sort_keys=True, default=str))
        return True
    if args.smoke_test:
        from maskattn_sdxl.smoke import run_tiny_gate_smoke

        print(run_tiny_gate_smoke())
        return True
    return False
