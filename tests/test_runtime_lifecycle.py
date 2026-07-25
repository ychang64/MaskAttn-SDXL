from __future__ import annotations

import pytest
import torch
from test_model import FakeUNet

from maskattn_sdxl.model import (
    MaskAttnConfig,
    assert_maskattn_device_consistency,
    assert_maskattn_ready,
    install_maskattn,
    load_maskattn_checkpoint,
    save_maskattn_checkpoint,
)
from maskattn_sdxl.runtime import load_maskattn_pipeline


class FakePipeline:
    def __init__(self) -> None:
        self.unet = FakeUNet()


def _checkpoint(tmp_path, config: MaskAttnConfig, *, global_step: int | None = 1):
    source = FakeUNet()
    installation = install_maskattn(source, config)
    return save_maskattn_checkpoint(source, tmp_path / "gates.pt", installation=installation, global_step=global_step)


def test_checkpoint_required_mode_fails_before_loading_a_pipeline() -> None:
    with pytest.raises(RuntimeError, match="requires `checkpoint`"):
        load_maskattn_pipeline(
            "unused",
            checkpoint=None,
            maskattn_config={"stage": "mid", "placement": "encoder_decoder"},
        )


def test_generation_loader_rejects_random_gate_inference() -> None:
    from maskattn_sdxl.generation import _load_pipeline_for_generation

    with pytest.raises(RuntimeError, match="requires `checkpoint`"):
        _load_pipeline_for_generation(
            {
                "method": "maskattn",
                "checkpoint": None,
                "maskattn": {"stage": "mid", "placement": "encoder_decoder"},
                "model": {"path": "unused"},
            }
        )


def test_checkpoint_metadata_attaches_to_the_final_pipeline_unet(monkeypatch, tmp_path) -> None:
    config = MaskAttnConfig(stage="high", placement="encoder", gate_hidden_dim=4)
    checkpoint = _checkpoint(tmp_path, config)
    pipe = FakePipeline()
    monkeypatch.setattr("maskattn_sdxl.runtime.load_sdxl_pipeline", lambda *args, **kwargs: pipe)

    loaded_pipe, installation, audit = load_maskattn_pipeline("unused", checkpoint=checkpoint, maskattn_config=None)

    assert loaded_pipe is pipe
    assert id(pipe.unet) == audit["final_unet_id"] == audit["installation_unet_id"] == audit["checkpoint_load_unet_id"]
    assert installation.config == config
    assert audit["trained_checkpoint_loaded"] is True
    assert audit["checkpoint_hash"]


def test_checkpoint_config_mismatch_and_wrong_unet_fail(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path, MaskAttnConfig(stage="high", placement="encoder", gate_hidden_dim=4))
    mismatched = FakeUNet()
    install_maskattn(mismatched, MaskAttnConfig(stage="high", placement="decoder", gate_hidden_dim=4))
    with pytest.raises(AssertionError, match="config differs"):
        load_maskattn_checkpoint(mismatched, checkpoint)

    with pytest.raises(AssertionError, match="not installed"):
        load_maskattn_checkpoint(FakeUNet(), checkpoint)


def test_gate_devices_follow_final_unet_for_both_install_orders() -> None:
    after_move = FakeUNet().to("cpu")
    install_maskattn(after_move, MaskAttnConfig(stage="mid", placement="encoder_decoder", gate_hidden_dim=4))
    assert_maskattn_device_consistency(after_move)

    before_move = FakeUNet()
    install_maskattn(before_move, MaskAttnConfig(stage="mid", placement="encoder_decoder", gate_hidden_dim=4))
    before_move.to("cpu")
    assert_maskattn_device_consistency(before_move)


def test_untrained_smoke_is_explicit_and_cannot_be_ready_as_trained(monkeypatch) -> None:
    pipe = FakePipeline()
    monkeypatch.setattr("maskattn_sdxl.runtime.load_sdxl_pipeline", lambda *args, **kwargs: pipe)
    _, _, audit = load_maskattn_pipeline(
        "unused",
        checkpoint=None,
        maskattn_config={"stage": "mid", "placement": "encoder_decoder", "gate_hidden_dim": 4},
        allow_untrained_gates=True,
    )
    assert audit["method"] == "UNTRAINED_INTEGRATION_ONLY"
    with pytest.raises(RuntimeError, match="requires a trained gate checkpoint"):
        assert_maskattn_ready(pipe.unet, checkpoint_required=True)


def test_test_only_checkpoint_cannot_be_used_for_inference(monkeypatch, tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path, MaskAttnConfig(stage="high", placement="encoder", gate_hidden_dim=4), global_step=None)
    monkeypatch.setattr("maskattn_sdxl.runtime.load_sdxl_pipeline", lambda *args, **kwargs: FakePipeline())
    with pytest.raises(RuntimeError, match="requires a trained gate checkpoint"):
        load_maskattn_pipeline("unused", checkpoint=checkpoint, maskattn_config=None)

    pipe, _, audit = load_maskattn_pipeline(
        "unused",
        checkpoint=checkpoint,
        maskattn_config=None,
        allow_test_only_checkpoint=True,
    )
    assert audit["method"] == "TEST_ONLY_UNTRAINED_CHECKPOINT"
    assert audit["trained_checkpoint_loaded"] is False
    with pytest.raises(RuntimeError, match="requires a trained gate checkpoint"):
        assert_maskattn_ready(pipe.unet, checkpoint_required=True)


def test_checkpoint_tensors_are_loaded_into_the_current_processors(tmp_path) -> None:
    config = MaskAttnConfig(stage="high", placement="encoder", gate_hidden_dim=4)
    checkpoint = _checkpoint(tmp_path, config)
    target = FakeUNet()
    install_maskattn(target, config)
    payload = load_maskattn_checkpoint(target, checkpoint)
    current = {key: value.detach().cpu() for key, value in target.state_dict().items() if ".processor.gate_head." in key}
    assert current.keys() == payload["gate_state_dict"].keys()
    assert all(torch.equal(current[key], payload["gate_state_dict"][key]) for key in current)
