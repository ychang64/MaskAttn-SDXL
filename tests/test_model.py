from __future__ import annotations

import torch
from torch import nn

from maskattn_sdxl.model import (
    MaskAttnConfig,
    assert_only_maskattn_trainable,
    install_maskattn,
    iter_maskattn_processors,
    load_maskattn_checkpoint,
    save_maskattn_checkpoint,
)


class FakeAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query_dim = channels
        self.cross_attention_dim = 10
        self.base = nn.Linear(channels, channels)
        self.processor = object()

    def set_processor(self, processor) -> None:
        self.processor = processor


class Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn2 = FakeAttention(channels)


class FakeUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList([Block(4), Block(4), Block(8)])
        self.up_blocks = nn.ModuleList([Block(8), Block(4), Block(4)])
        self.mid_block = Block(8)

    @property
    def attn_processors(self):
        return {
            "down_blocks.1.attn2.processor": self.down_blocks[1].attn2.processor,
            "down_blocks.2.attn2.processor": self.down_blocks[2].attn2.processor,
            "up_blocks.0.attn2.processor": self.up_blocks[0].attn2.processor,
            "up_blocks.1.attn2.processor": self.up_blocks[1].attn2.processor,
            "mid_block.attn2.processor": self.mid_block.attn2.processor,
        }

    def set_attn_processor(self, processors) -> None:
        for name, processor in processors.items():
            self.get_submodule(name.removesuffix(".processor")).set_processor(processor)


def test_stage_and_placement_select_expected_modules() -> None:
    unet = FakeUNet()
    installation = install_maskattn(unet, MaskAttnConfig(stage="mid", placement="encoder_decoder", gate_hidden_dim=4))
    assert installation.selected_layers == ["down_blocks.2.attn2.processor", "up_blocks.0.attn2.processor"]
    assert {name for name, _ in iter_maskattn_processors(unet)} == set(installation.selected_layers)
    assert_only_maskattn_trainable(unet)


def test_gate_checkpoint_round_trip(tmp_path) -> None:
    unet = FakeUNet()
    installation = install_maskattn(unet, MaskAttnConfig(stage="high", placement="encoder", gate_hidden_dim=4))
    checkpoint = save_maskattn_checkpoint(unet, tmp_path / "gates.pt", installation=installation, global_step=7)
    restored = FakeUNet()
    install_maskattn(restored, MaskAttnConfig(stage="high", placement="encoder", gate_hidden_dim=4))
    payload = load_maskattn_checkpoint(restored, checkpoint)
    assert payload["global_step"] == 7
    source_values = [value.detach().clone() for _, processor in iter_maskattn_processors(unet) for value in processor.gate_head.parameters()]
    target_values = [value.detach().clone() for _, processor in iter_maskattn_processors(restored) for value in processor.gate_head.parameters()]
    assert len(source_values) == len(target_values)
    assert all(torch.equal(source, target) for source, target in zip(source_values, target_values))
