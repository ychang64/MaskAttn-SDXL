"""Real metric adapters; unavailable dependencies raise actionable errors instead of fabricating scores."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _image_paths(directory: str | Path) -> list[Path]:
    root = Path(directory)
    paths = sorted(path for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp") for path in root.glob(suffix))
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths


def compute_fid_and_precision_recall(
    generated_dir: str | Path, real_dir: str | Path, *, batch_size: int = 16, device: str = "cuda"
) -> dict[str, float]:
    """Use Clean-FID features and k-NN manifold precision/recall (Kynkäänniemi-style)."""
    try:
        import numpy as np
        from cleanfid import fid
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError(
            "FID/precision/recall requires optional dependencies. Install `pip install clean-fid scikit-learn`."
        ) from exc

    generated_paths = _image_paths(generated_dir)
    real_paths = _image_paths(real_dir)
    # Clean-FID's public helper extracts Inception features; it does not synthesize reference statistics.
    generated_features = fid.get_folder_features(str(generated_dir), model_name="inception_v3", batch_size=batch_size, device=device)
    real_features = fid.get_folder_features(str(real_dir), model_name="inception_v3", batch_size=batch_size, device=device)
    fid_value = float(fid.frechet_distance(real_features, generated_features))

    def radii(features: np.ndarray, k: int = 3) -> np.ndarray:
        neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(features))).fit(features)
        distances, _ = neighbors.kneighbors(features)
        return distances[:, -1]

    def manifold_coverage(query: np.ndarray, reference: np.ndarray, reference_radii: np.ndarray) -> float:
        nearest = NearestNeighbors(n_neighbors=1).fit(reference)
        distances, indices = nearest.kneighbors(query)
        return float((distances[:, 0] <= reference_radii[indices[:, 0]]).mean())

    if min(len(real_features), len(generated_features)) < 4:
        raise ValueError("Precision/recall requires at least four real and generated images.")
    precision = manifold_coverage(generated_features, real_features, radii(real_features))
    recall = manifold_coverage(real_features, generated_features, radii(generated_features))
    return {
        "fid": fid_value,
        "precision": precision,
        "recall": recall,
        "num_generated": len(generated_paths),
        "num_real": len(real_paths),
    }


def compute_clip_score(images_dir: str | Path, prompts: Iterable[str], *, device: str = "cuda") -> float:
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError as exc:
        raise ImportError("CLIP Score requires `open-clip-torch`. Install `pip install open-clip-torch`.") from exc
    paths = _image_paths(images_dir)
    prompts = list(prompts)
    if len(paths) != len(prompts):
        raise ValueError(f"CLIP Score needs one prompt per image; got {len(paths)} images and {len(prompts)} prompts")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for image_path, prompt in zip(paths, prompts):
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            text = tokenizer([prompt]).to(device)
            image_features = torch.nn.functional.normalize(model.encode_image(image), dim=-1)
            text_features = torch.nn.functional.normalize(model.encode_text(text), dim=-1)
            scores.append(float((image_features * text_features).sum().cpu()))
    return sum(scores) / len(scores)
