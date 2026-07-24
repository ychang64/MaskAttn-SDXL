"""Integration tests against the public Diffusers Attention implementation."""

from __future__ import annotations

import copy

import pytest
import torch

from maskattn_sdxl.gating import MaskGateHead
from maskattn_sdxl.processor import MaskAttnProcessor


def _all_open_gate() -> MaskGateHead:
    gate = MaskGateHead(feature_dim=4, token_dim=6, hidden_dim=4)
    with torch.no_grad():
        for parameter in gate.parameters():
            parameter.zero_()
        gate.logit_bias.fill_(20.0)
    return gate


def test_all_open_gate_matches_real_diffusers_attention() -> None:
    """A hard all-open gate is an exact zero bias and must preserve base attention."""
    pytest.importorskip("diffusers")
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0

    torch.manual_seed(7)
    reference = Attention(query_dim=4, cross_attention_dim=6, heads=2, dim_head=2, bias=True, out_bias=True)
    masked = copy.deepcopy(reference)
    reference.set_processor(AttnProcessor2_0())
    masked.set_processor(MaskAttnProcessor(_all_open_gate(), threshold=0.5))
    hidden_states = torch.randn(2, 5, 4)
    encoder_hidden_states = torch.randn(2, 7, 6)

    expected = reference(hidden_states, encoder_hidden_states=encoder_hidden_states)
    actual = masked(hidden_states, encoder_hidden_states=encoder_hidden_states)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
