from __future__ import annotations

import torch
import torch.nn.functional as F

from maskattn_sdxl.gating import MaskGateHead, hard_gate_with_ste, masked_attention


def test_gate_probability_shape_and_range() -> None:
    head = MaskGateHead(feature_dim=6, token_dim=8, hidden_dim=4)
    probabilities = head(torch.randn(2, 5, 6), torch.randn(2, 7, 8))
    assert probabilities.shape == (2, 5, 7)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_new_gate_head_is_open_everywhere() -> None:
    head = MaskGateHead(feature_dim=1280, token_dim=2048)
    probabilities = head(torch.randn(2, 256, 1280) * 5, torch.randn(2, 77, 2048) * 5)
    assert torch.all(hard_gate_with_ste(probabilities).hard_gate == 1)


def test_hard_gate_and_additive_mask_are_exact() -> None:
    probabilities = torch.tensor([[[0.2, 0.8, 0.1]]], requires_grad=True)
    result = hard_gate_with_ste(probabilities, threshold=0.5, negative_value=-10000.0)
    assert torch.equal(result.hard_gate, torch.tensor([[[0.0, 1.0, 0.0]]]))
    assert torch.equal(result.additive_mask, torch.tensor([[[-10000.0, 0.0, -10000.0]]]))
    assert not result.additive_mask.requires_grad


def test_all_masked_rows_retain_a_fallback_and_do_not_nan() -> None:
    probabilities = torch.full((2, 3, 4), 0.1, requires_grad=True)
    result = hard_gate_with_ste(probabilities, threshold=1.0)
    assert torch.equal(result.hard_gate.sum(dim=-1), torch.ones(2, 3))
    query, key, value = torch.randn(2, 1, 3, 2), torch.randn(2, 1, 4, 2), torch.randn(2, 1, 4, 2)
    output, _ = masked_attention(query, key, value, result)
    assert torch.isfinite(output).all()


def test_masked_attention_is_eq_4() -> None:
    query, key, value = torch.randn(2, 2, 5, 8), torch.randn(2, 2, 7, 8), torch.randn(2, 2, 7, 8)
    gate = hard_gate_with_ste(torch.rand(2, 5, 7))
    actual, _ = masked_attention(query, key, value, gate)
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=gate.additive_mask[:, None])
    torch.testing.assert_close(actual, expected)


def test_training_forward_value_is_the_hard_attention() -> None:
    query, key, value = torch.randn(1, 1, 3, 4), torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4)
    probabilities = torch.rand(1, 3, 5)
    with torch.no_grad():
        inference, _ = masked_attention(query, key, value, hard_gate_with_ste(probabilities))
    training, _ = masked_attention(query, key, value, hard_gate_with_ste(probabilities.requires_grad_()))
    assert torch.equal(inference, training.detach())


def test_masked_token_contributes_nothing() -> None:
    query = torch.tensor([[[[1.0]]]])
    key = torch.tensor([[[[1.0], [1.0]]]])
    value = torch.tensor([[[[0.0], [100.0]]]])
    gate = hard_gate_with_ste(torch.tensor([[[0.9, 0.1]]]))
    output, weights = masked_attention(query, key, value, gate)
    assert float(output) == 0.0
    assert torch.equal(weights, torch.tensor([[[[1.0, 0.0]]]]))


def test_gradient_reaches_closed_gates_and_stays_bounded() -> None:
    probabilities = torch.tensor([[[0.9, 0.2, 0.7]]], requires_grad=True)
    query, key, value = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 3, 8), torch.randn(1, 1, 3, 8)
    output, _ = masked_attention(query, key, value, hard_gate_with_ste(probabilities))
    output.sum().backward()
    assert float(probabilities.grad[0, 0, 1].abs()) > 0
    assert float(probabilities.grad.abs().max()) < 100


def test_closed_gate_reopens_when_the_token_is_needed() -> None:
    gate_logits = torch.nn.Parameter(torch.tensor([[[-2.0, 2.0]]]))
    query = torch.ones(1, 1, 1, 1)
    key = torch.ones(1, 1, 2, 1)
    value = torch.tensor([[[[1.0], [0.0]]]])
    optimizer = torch.optim.SGD([gate_logits], lr=5.0)
    for _ in range(100):
        output, _ = masked_attention(query, key, value, hard_gate_with_ste(torch.sigmoid(gate_logits)))
        loss = (output - 1.0).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert gate_logits[0, 0, 0].item() > 0
