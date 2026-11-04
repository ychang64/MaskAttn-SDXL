"""Integration tests against the public Diffusers Attention implementation and SDXL's real topology."""

from __future__ import annotations

import copy

import pytest
import torch

from maskattn_sdxl.gating import MaskGateHead
from maskattn_sdxl.model import MaskAttnConfig, gate_heads, install_maskattn, maskattn_parameters
from maskattn_sdxl.processor import MaskAttnProcessor


def _attention_pair():
    pytest.importorskip("diffusers")
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0

    torch.manual_seed(7)
    reference = Attention(query_dim=4, cross_attention_dim=6, heads=2, dim_head=2, bias=True, out_bias=True)
    masked = copy.deepcopy(reference)
    reference.set_processor(AttnProcessor2_0())
    return reference, masked


def test_all_open_gate_matches_real_diffusers_attention() -> None:
    """A new gate is open everywhere, so an untrained model must reproduce the base attention."""
    reference, masked = _attention_pair()
    masked.set_processor(MaskAttnProcessor(MaskGateHead(feature_dim=4, token_dim=6, hidden_dim=4)))
    hidden_states = torch.randn(2, 5, 4)
    encoder_hidden_states = torch.randn(2, 7, 6)

    expected = reference(hidden_states, encoder_hidden_states=encoder_hidden_states)
    actual = masked(hidden_states, encoder_hidden_states=encoder_hidden_states)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert masked.processor.forward_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuDNN regression needs a GPU")
def test_gate_training_backward_runs_on_cuda() -> None:
    """cuDNN's SDPA backward rejects an attn_mask with gradients, which is how these gates train."""
    _, masked = _attention_pair()
    masked = masked.to("cuda")
    gate = MaskGateHead(feature_dim=4, token_dim=6, hidden_dim=4).to("cuda")
    masked.set_processor(MaskAttnProcessor(gate).to("cuda"))
    output = masked(torch.randn(2, 5, 4, device="cuda"), encoder_hidden_states=torch.randn(2, 7, 6, device="cuda"))
    output.square().mean().backward()
    assert gate.logit_bias.grad is not None


@pytest.mark.parametrize(
    ("stage", "sites"),
    [("high", 10), ("mid", 50), ("low", 10), ("all", 70)],
)
def test_real_sdxl_topology_stage_selection(stage: str, sites: int) -> None:
    pytest.importorskip("diffusers")
    from tiny_sdxl import build_sdxl_unet

    with torch.device("meta"):
        unet = build_sdxl_unet()
        installation = install_maskattn(unet, MaskAttnConfig(stage=stage))
    assert len(installation.selected_layers) == sites


def test_default_gates_add_the_parameters_reported_in_the_paper() -> None:
    pytest.importorskip("diffusers")
    from tiny_sdxl import build_sdxl_unet

    with torch.device("meta"):
        unet = build_sdxl_unet()
        install_maskattn(unet, MaskAttnConfig())
    assert len(gate_heads(unet)) == 5
    assert sum(parameter.numel() for parameter in maskattn_parameters(unet)) == 3_600_645
