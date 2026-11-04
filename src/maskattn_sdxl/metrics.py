"""Real metric adapters; unavailable dependencies raise actionable errors instead of fabricating scores."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np


def _image_paths(directory: str | Path) -> list[Path]:
    root = Path(directory)
    paths = sorted(path for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp") for path in root.glob(suffix))
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths


def frechet_distance_from_features(real: np.ndarray, generated: np.ndarray) -> float:
    """||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2)).

    Tr((S_r S_g)^(1/2)) equals Tr((S_r^(1/2) S_g S_r^(1/2))^(1/2)), whose argument is symmetric. Eigendecomposition
    is therefore stable even when the covariances are rank-deficient, which they are below 2048 images. It also
    avoids clean-fid's own helper, which passes a `disp` argument that current SciPy has removed.
    """
    mean_difference = np.mean(real, axis=0) - np.mean(generated, axis=0)
    covariance_real = np.cov(real, rowvar=False)
    covariance_generated = np.cov(generated, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_real)
    root = (eigenvectors * np.sqrt(eigenvalues.clip(min=0.0))) @ eigenvectors.T
    trace_of_root = np.sqrt(np.linalg.eigvalsh(root @ covariance_generated @ root).clip(min=0.0)).sum()
    return float(mean_difference @ mean_difference + np.trace(covariance_real) + np.trace(covariance_generated) - 2.0 * trace_of_root)


def _knn_radii(features: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    # The nearest neighbour of a point in its own set is itself, hence k + 1.
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(features).kneighbors(features)
    return distances[:, -1]


def _manifold_coverage(query: np.ndarray, reference: np.ndarray, radii: np.ndarray, *, chunk_size: int = 1024) -> float:
    """Fraction of query points inside at least one reference point's k-NN ball (Kynkaanniemi et al., 2019)."""
    from scipy.spatial.distance import cdist

    covered = 0
    for start in range(0, len(query), chunk_size):
        distances = cdist(query[start : start + chunk_size], reference)
        covered += int((distances <= radii[None, :]).any(axis=1).sum())
    return covered / len(query)


def precision_recall_from_features(real: np.ndarray, generated: np.ndarray, *, k: int = 3) -> tuple[float, float]:
    if min(len(real), len(generated)) <= k:
        raise ValueError(f"Precision/recall requires more than {k} real and generated images.")
    precision = _manifold_coverage(generated, real, _knn_radii(real, k))
    recall = _manifold_coverage(real, generated, _knn_radii(generated, k))
    return precision, recall


def compute_fid_and_precision_recall(
    generated_dir: str | Path, real_dir: str | Path, *, batch_size: int = 16, device: str = "cuda"
) -> dict[str, float]:
    """Compute Clean-FID, then manifold precision and recall on the same Inception features."""
    try:
        import torch
        from cleanfid import fid
    except ImportError as exc:
        raise ImportError(
            "FID/precision/recall requires optional dependencies. Install `pip install -e .[metrics]`."
        ) from exc

    generated_paths = _image_paths(generated_dir)
    real_paths = _image_paths(real_dir)
    torch_device = torch.device(device)
    extractor = fid.build_feature_extractor("clean", torch_device)
    # Clean-FID's resize closures cannot be pickled to spawned workers on Windows.
    num_workers = 0 if os.name == "nt" else 8

    def features(directory: str | Path) -> np.ndarray:
        return fid.get_folder_features(
            str(directory), model=extractor, num_workers=num_workers, batch_size=batch_size, device=torch_device, mode="clean"
        )

    generated_features = features(generated_dir)
    real_features = features(real_dir)
    precision, recall = precision_recall_from_features(real_features, generated_features)
    return {
        "fid": frechet_distance_from_features(real_features, generated_features),
        "precision": precision,
        "recall": recall,
        "num_generated": len(generated_paths),
        "num_real": len(real_paths),
    }


def compute_clip_score(
    images_dir: str | Path,
    prompts: Iterable[str],
    *,
    device: str = "cuda",
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
) -> float:
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError as exc:
        raise ImportError("CLIP Score requires `open-clip-torch`. Install `pip install -e .[metrics]`.") from exc
    paths = _image_paths(images_dir)
    prompts = list(prompts)
    if len(paths) != len(prompts):
        raise ValueError(f"CLIP Score needs one prompt per image; got {len(paths)} images and {len(prompts)} prompts")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
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
