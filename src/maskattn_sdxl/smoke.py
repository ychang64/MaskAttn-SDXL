"""CPU-sized checks used by experiment ``--smoke-test`` modes."""

from __future__ import annotations


def run_tiny_gate_smoke() -> dict[str, object]:
    import torch
    import torch.nn.functional as F

    from .gating import MaskGateHead, hard_gate_with_ste

    torch.manual_seed(0)
    head = MaskGateHead(feature_dim=8, token_dim=12, hidden_dim=4)
    features = torch.randn(2, 5, 8, requires_grad=True)
    tokens = torch.randn(2, 4, 12)
    probabilities = head(features, tokens)
    result = hard_gate_with_ste(probabilities, threshold=1.0, negative_value=-10000.0)
    # Threshold one forces the safe fallback branch for every location.
    query = torch.randn(2, 1, 5, 3)
    key = torch.randn(2, 1, 4, 3)
    values = torch.randn(2, 1, 4, 3)
    output = F.scaled_dot_product_attention(query, key, values, attn_mask=result.additive_mask[:, None])
    loss = output.square().mean() + result.ste_gate.mean()
    loss.backward()
    if not torch.isfinite(output).all() or not torch.isfinite(features.grad).all():
        raise AssertionError("Tiny gate smoke produced NaN/Inf")
    return {"probability_shape": list(probabilities.shape), "output_shape": list(output.shape), "finite": True}
