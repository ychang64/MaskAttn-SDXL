from __future__ import annotations

import torch
import torch.nn.functional as F

from maskattn_sdxl.gating import MaskGateHead, hard_gate_with_ste


def test_gate_probability_shape_and_range() -> None:
    head = MaskGateHead(feature_dim=6, token_dim=8, hidden_dim=4)
    probabilities = head(torch.randn(2, 5, 6), torch.randn(2, 7, 8))
    assert probabilities.shape == (2, 5, 7)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_hard_gate_is_binary_and_ste_backpropagates() -> None:
    probabilities = torch.tensor([[[0.2, 0.8, 0.1]]], requires_grad=True)
    result = hard_gate_with_ste(probabilities, threshold=0.5)
    assert torch.equal(result.hard_gate, torch.tensor([[[0.0, 1.0, 0.0]]]))
    result.ste_gate.sum().backward()
    assert probabilities.grad is not None
    assert torch.count_nonzero(probabilities.grad) > 0


def test_all_masked_rows_retain_a_fallback_and_do_not_nan() -> None:
    probabilities = torch.full((2, 3, 4), 0.1, requires_grad=True)
    result = hard_gate_with_ste(probabilities, threshold=1.0)
    assert torch.equal(result.hard_gate.sum(dim=-1), torch.ones(2, 3))
    query = torch.randn(2, 1, 3, 2)
    key = torch.randn(2, 1, 4, 2)
    value = torch.randn(2, 1, 4, 2)
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=result.additive_mask[:, None])
    assert torch.isfinite(output).all()


def test_additive_mask_changes_logits_before_softmax() -> None:
    # Token 1 carries a very large value. A large negative pre-softmax bias must remove its contribution.
    query = torch.tensor([[[[1.0]]]])
    key = torch.tensor([[[[1.0], [1.0]]]])
    value = torch.tensor([[[[0.0], [100.0]]]])
    no_mask = F.scaled_dot_product_attention(query, key, value)
    additive = torch.tensor([[[[0.0, -10000.0]]]])
    masked = F.scaled_dot_product_attention(query, key, value, attn_mask=additive)
    assert float(no_mask) > 40.0
    assert float(masked) == 0.0
