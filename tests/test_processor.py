from __future__ import annotations

import torch
from torch import nn

from maskattn_sdxl.gating import MaskGateHead
from maskattn_sdxl.processor import MaskAttnProcessor


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_dim = 4
        self.cross_attention_dim = 6
        self.heads = 2
        self.to_q = nn.Linear(4, 4, bias=False)
        self.to_k = nn.Linear(6, 4, bias=False)
        self.to_v = nn.Linear(6, 4, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(4, 4, bias=False), nn.Identity()])
        self.group_norm = None
        self.spatial_norm = None
        self.norm_cross = None
        self.norm_q = None
        self.norm_k = None
        self.residual_connection = False
        self.rescale_output_factor = 1.0

    def prepare_attention_mask(self, mask, sequence_length, batch_size):
        return mask


def test_processor_forward_backward_and_mask_capture() -> None:
    attention = TinyAttention()
    processor = MaskAttnProcessor(MaskGateHead(4, 6, 3), threshold=1.0)
    processor.set_record_masks(True)
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    text = torch.randn(2, 3, 6)
    output = processor(attention, hidden, encoder_hidden_states=text)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert processor.gate_head.feature_net[1].weight.grad is not None
    assert processor.last_gate is not None
