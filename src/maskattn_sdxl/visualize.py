"""Save token-wise gate artifacts for qualitative reproducibility."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _shape_from_locations(locations: int) -> tuple[int, int]:
    root = int(math.sqrt(locations))
    for height in range(root, 0, -1):
        if locations % height == 0:
            return height, locations // height
    return 1, locations


def save_token_masks(
    masks: Mapping[str, Mapping[str, object]],
    output_dir: str | Path,
    *,
    sample_name: str,
    token_labels: Sequence[str] | None = None,
) -> Path:
    """Persist exact flattened masks and readable PNGs when PIL is available."""
    import torch

    root = Path(output_dir) / sample_name
    root.mkdir(parents=True, exist_ok=True)
    torch.save(dict(masks), root / "masks.pt")
    try:
        from PIL import Image
    except ImportError:
        return root

    for layer, values in masks.items():
        probabilities = values["probabilities"]
        hard_gate = values["hard_gate"]
        # Qualitative runs conventionally use batch size one; retain only the first item for PNG rendering.
        attention = values.get("attention")
        render_values = [("prob", probabilities), ("hard", hard_gate)]
        if attention is not None:
            render_values.append(("attention", attention))
        for value_name, tensor in render_values:
            array = tensor[0].numpy()  # [N, T]
            height, width = _shape_from_locations(array.shape[0])
            layer_dir = root / layer.replace(".", "_") / value_name
            layer_dir.mkdir(parents=True, exist_ok=True)
            for token_index in range(array.shape[1]):
                image = array[:, token_index].reshape(height, width)
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
                label = token_labels[token_index] if token_labels and token_index < len(token_labels) else f"token_{token_index:03d}"
                safe_label = "".join(char if char.isalnum() else "_" for char in label)[:40]
                Image.fromarray(image, mode="L").save(layer_dir / f"{token_index:03d}_{safe_label}.png")
    return root
