"""CPU-sized checks used by experiment ``--smoke-test`` modes."""

from __future__ import annotations


def run_tiny_gate_smoke() -> dict[str, object]:
    import torch

    from .gating import MaskGateHead, hard_gate_with_ste, masked_attention

    torch.manual_seed(0)
    head = MaskGateHead(feature_dim=8, token_dim=12, hidden_dim=4)
    features = torch.randn(2, 5, 8)
    tokens = torch.randn(2, 4, 12)
    probabilities = head(features, tokens)
    # Threshold one forces the safe fallback branch for every location.
    gate = hard_gate_with_ste(probabilities, threshold=1.0, negative_value=-10000.0)
    query = torch.randn(2, 1, 5, 3)
    key = torch.randn(2, 1, 4, 3)
    values = torch.randn(2, 1, 4, 3)
    output, _ = masked_attention(query, key, values, gate)
    output.square().mean().backward()
    gradients = [parameter.grad for parameter in head.parameters()]
    if not torch.isfinite(output).all() or any(grad is None or not torch.isfinite(grad).all() for grad in gradients):
        raise AssertionError("Tiny gate smoke produced NaN/Inf or a missing gradient")
    return {"probability_shape": list(probabilities.shape), "output_shape": list(output.shape), "finite": True}
