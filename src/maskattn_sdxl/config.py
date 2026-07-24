"""Small configuration helpers shared by every experiment entry point."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into a copy of ``base``."""
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(dict(result[key]), value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False, allow_unicode=True)


def get_by_dotted_key(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    current: Any = config
    for piece in key.split("."):
        if not isinstance(current, Mapping) or piece not in current:
            return default
        current = current[piece]
    return current


def set_by_dotted_key(config: dict[str, Any], key: str, value: Any) -> None:
    current = config
    pieces = key.split(".")
    for piece in pieces[:-1]:
        current = current.setdefault(piece, {})
        if not isinstance(current, dict):
            raise ValueError(f"Cannot assign {key}: {piece} is not a mapping")
    current[pieces[-1]] = value


def parse_overrides(values: list[str] | None) -> dict[str, Any]:
    """Parse ``key=value`` CLI overrides using YAML scalar semantics."""
    result: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Expected key=value override, received: {item}")
        key, raw_value = item.split("=", 1)
        if not key:
            raise ValueError(f"Override key is empty: {item}")
        set_by_dotted_key(result, key, yaml.safe_load(raw_value))
    return result


def git_commit(root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(root: str | Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(root),
    }
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
            }
        )
    except Exception as exc:  # Environment reporting must not break an experiment.
        metadata["torch_import_error"] = repr(exc)
    return metadata


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        # NumPy is a runtime dependency for training/metrics, but config-only modes remain lightweight.
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def prepare_run_directory(output_dir: str | Path, config: Mapping[str, Any], root: str | Path) -> Path:
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config.resolved.yaml")
    with (run_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment_metadata(root), handle, indent=2, sort_keys=True)
    return run_dir
