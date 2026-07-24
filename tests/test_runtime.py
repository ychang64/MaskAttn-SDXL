from __future__ import annotations

import pytest

from maskattn_sdxl.runtime import resolve_model_source


def test_hub_model_id_requires_explicit_download_permission() -> None:
    with pytest.raises(FileNotFoundError, match="allow_download=true"):
        resolve_model_source("organization/model", allow_download=False)
    assert resolve_model_source("organization/model", allow_download=True) == ("organization/model", False)
