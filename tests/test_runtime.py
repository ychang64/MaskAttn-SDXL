from __future__ import annotations

import pytest

from maskattn_sdxl.runtime import resolve_model_source, weights_variant


def test_weights_variant_follows_the_files_on_disk(tmp_path) -> None:
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "diffusion_pytorch_model.safetensors").touch()
    assert weights_variant(tmp_path) is None
    (tmp_path / "unet" / "diffusion_pytorch_model.fp16.safetensors").touch()
    assert weights_variant(tmp_path) == "fp16"


def test_hub_model_id_requires_explicit_download_permission() -> None:
    with pytest.raises(FileNotFoundError, match="allow_download=true"):
        resolve_model_source("organization/model", allow_download=False)
    assert resolve_model_source("organization/model", allow_download=True) == ("organization/model", False)
