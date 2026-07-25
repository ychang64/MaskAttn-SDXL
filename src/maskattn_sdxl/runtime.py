"""Runtime loading helpers. Imports of heavy ML dependencies happen only when an experiment runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str):
    import torch

    if name == "auto":
        return torch.float16 if device in {"cuda", "mps"} else torch.float32
    choices = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in choices:
        raise ValueError(f"Unsupported dtype {name}; choose auto, float16, bfloat16, or float32")
    return choices[name]


def require_local_model(model_path: str | Path) -> Path:
    path = Path(model_path)
    required = ["model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors"]
    missing = [str(path / item) for item in required if not (path / item).is_file()]
    if missing:
        raise FileNotFoundError(
            "SDXL model is incomplete. Expected a Diffusers-format local model directory. Missing: " + ", ".join(missing)
        )
    return path


def resolve_model_source(model_source: str | Path, *, allow_download: bool) -> tuple[str, bool]:
    """Resolve a local Diffusers directory or an explicit Hugging Face repository ID.

    A non-existent path is accepted only as a Hub ID and only when the caller has
    explicitly allowed a download.  This prevents experiments from silently
    fetching multi-gigabyte SDXL weights.
    """
    candidate = Path(model_source)
    if candidate.exists():
        return str(require_local_model(candidate)), True
    source = str(model_source)
    if "/" not in source:
        raise FileNotFoundError(
            f"Model source does not exist: {source}. Provide a local Diffusers directory or a Hugging Face repo ID "
            "such as stabilityai/stable-diffusion-xl-base-1.0."
        )
    if not allow_download:
        raise FileNotFoundError(
            f"Model {source!r} is not cached locally. To permit a Hugging Face download, set model.allow_download=true; "
            "otherwise download it first and set model.path to its local directory."
        )
    return source, False


def load_sdxl_pipeline(
    model_path: str | Path,
    *,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = True,
    allow_download: bool = False,
    disable_safety_checker: bool = False,
):
    """Load a local SDXL model, or an explicitly authorised Hugging Face model ID."""
    try:
        from diffusers import StableDiffusionXLPipeline
    except ImportError as exc:
        raise ImportError(
            "Diffusers is unavailable. Install the pinned release with `pip install -e .` (or the local audit checkout)."
        ) from exc

    source, is_local = resolve_model_source(model_path, allow_download=allow_download)
    resolved_device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, resolved_device)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "variant": "fp16",
        "use_safetensors": True,
        "local_files_only": local_files_only if is_local else False,
    }
    if disable_safety_checker:
        kwargs["safety_checker"] = None
    pipe = StableDiffusionXLPipeline.from_pretrained(source, **kwargs)
    pipe.to(resolved_device)
    return pipe


def checkpoint_maskattn_config(checkpoint_path: str | Path):
    """Read the authoritative MaskAttn installation configuration from a gate checkpoint."""
    import torch

    from .model import MaskAttnConfig

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"MaskAttn checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "maskattn-sdxl-gates-v1":
        raise ValueError(f"Not a MaskAttn-SDXL gate checkpoint: {path}")
    return MaskAttnConfig.from_mapping(payload["maskattn_config"])


def checkpoint_maskattn_kind(checkpoint_path: str | Path) -> str:
    """Read checkpoint provenance before allocating a pipeline for inference."""
    import torch

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"MaskAttn checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "maskattn-sdxl-gates-v1":
        raise ValueError(f"Not a MaskAttn-SDXL gate checkpoint: {path}")
    kind = payload.get("checkpoint_kind", "trained" if payload.get("global_step") is not None else "test_only")
    if kind not in {"trained", "test_only"}:
        raise ValueError(f"Unsupported MaskAttn checkpoint kind: {kind}")
    return kind


def load_maskattn_pipeline(
    model_path: str | Path,
    *,
    checkpoint: str | Path | None,
    maskattn_config: dict[str, Any] | None = None,
    allow_untrained_gates: bool = False,
    allow_test_only_checkpoint: bool = False,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = True,
    allow_download: bool = False,
    disable_safety_checker: bool = False,
):
    """Load one final SDXL pipeline and attach verified MaskAttn processors to its U-Net.

    Checkpoint metadata is the inference configuration source. A caller-supplied config is only accepted when it
    exactly matches that metadata; this prevents loading gate weights into a differently wired U-Net.
    """
    from .model import (
        MaskAttnConfig,
        assert_maskattn_ready,
        install_maskattn,
        load_maskattn_checkpoint,
        reset_maskattn_forward_calls,
    )

    if checkpoint is None and not allow_untrained_gates:
        raise RuntimeError(
            "MaskAttn-SDXL inference requires `checkpoint`. Random gates are only allowed for an explicit integration smoke."
        )
    checkpoint_kind: str | None = None
    if checkpoint is not None:
        checkpoint_kind = checkpoint_maskattn_kind(checkpoint)
        if checkpoint_kind != "trained" and not allow_test_only_checkpoint:
            raise RuntimeError("MaskAttn-SDXL inference requires a trained gate checkpoint; test-only checkpoints are rejected.")
        checkpoint_config = checkpoint_maskattn_config(checkpoint)
        if maskattn_config is not None and MaskAttnConfig.from_mapping(maskattn_config) != checkpoint_config:
            raise ValueError(
                "Configured MaskAttn stage/placement/gate architecture differs from checkpoint metadata. "
                "Remove the override or use a matching checkpoint."
            )
        config = checkpoint_config
    else:
        if maskattn_config is None:
            raise ValueError("Untrained MaskAttn smoke requires an explicit maskattn configuration")
        config = MaskAttnConfig.from_mapping(maskattn_config)

    pipe = load_sdxl_pipeline(
        model_path,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
        allow_download=allow_download,
        disable_safety_checker=disable_safety_checker,
    )
    final_unet = pipe.unet
    installation = install_maskattn(final_unet, config)
    if checkpoint is not None:
        load_maskattn_checkpoint(final_unet, checkpoint)
    else:
        final_unet._maskattn_runtime["allow_untrained_gates"] = True  # type: ignore[attr-defined]
    final_unet.eval()
    reset_maskattn_forward_calls(final_unet)
    audit = assert_maskattn_ready(final_unet, checkpoint_required=checkpoint_kind == "trained", expected_config=config)
    audit.update(
        {
            "method": "maskattn_sdxl"
            if checkpoint_kind == "trained"
            else "TEST_ONLY_UNTRAINED_CHECKPOINT"
            if checkpoint_kind == "test_only"
            else "UNTRAINED_INTEGRATION_ONLY",
            "final_unet_id": id(final_unet),
            "installation_unet_id": id(final_unet),
            "checkpoint_load_unet_id": id(final_unet) if checkpoint is not None else None,
        }
    )
    return pipe, installation, audit


def pipeline_components_frozen(pipe) -> None:
    for module_name in ("vae", "text_encoder", "text_encoder_2"):
        module = getattr(pipe, module_name, None)
        if module is not None:
            module.requires_grad_(False)
            module.eval()
