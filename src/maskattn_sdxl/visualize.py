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
    """Persist exact flattened masks, plus one PNG per labelled token.

    Under classifier-free guidance the batch is [unconditional, conditional], so the PNGs use the last item:
    the unconditional branch has no prompt tokens to show. `token_labels` must include BOS, as `prompt_token_labels`
    does, so that each label lines up with its position on the token axis.
    """
    import torch

    root = Path(output_dir) / sample_name
    root.mkdir(parents=True, exist_ok=True)
    torch.save(dict(masks), root / "masks.pt")
    try:
        from PIL import Image
    except ImportError:
        return root

    for layer, values in masks.items():
        render_values = [("prob", values["probabilities"]), ("hard", values["hard_gate"])]
        if values.get("attention") is not None:
            render_values.append(("attention", values["attention"]))
        for value_name, tensor in render_values:
            array = tensor[-1].numpy()  # [N, T]
            height, width = _shape_from_locations(array.shape[0])
            layer_dir = root / layer.replace(".", "_") / value_name
            layer_dir.mkdir(parents=True, exist_ok=True)
            token_count = min(len(token_labels), array.shape[1]) if token_labels else array.shape[1]
            for token_index in range(token_count):
                image = array[:, token_index].reshape(height, width)
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
                label = token_labels[token_index] if token_labels else f"token_{token_index:03d}"
                safe_label = "".join(char if char.isalnum() else "_" for char in label)[:40]
                Image.fromarray(image).save(layer_dir / f"{token_index:03d}_{safe_label}.png")
    return root
