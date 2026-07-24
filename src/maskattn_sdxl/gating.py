"""Token-conditioned gate heads and straight-through hard masking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class GateResult:
    probabilities: Tensor  # [B, N, T]
    hard_gate: Tensor  # [B, N, T], exact 0/1 forward values
    ste_gate: Tensor  # [B, N, T], hard forward / soft backward
    additive_mask: Tensor  # [B, N, T]


def _normalise_valid_tokens(valid_tokens: Optional[Tensor], batch: int, tokens: int, device: torch.device) -> Tensor:
    if valid_tokens is None:
        return torch.ones((batch, tokens), dtype=torch.bool, device=device)
    if valid_tokens.ndim != 2 or valid_tokens.shape != (batch, tokens):
        raise ValueError(
            f"valid_tokens must have shape {(batch, tokens)}, received {tuple(valid_tokens.shape)}"
        )
    valid_tokens = valid_tokens.to(device=device, dtype=torch.bool)
    if not bool(valid_tokens.any(dim=-1).all()):
        raise ValueError("Every batch item needs at least one valid text token for MaskAttn.")
    return valid_tokens


def hard_gate_with_ste(
    probabilities: Tensor,
    *,
    threshold: float = 0.5,
    negative_value: float = -1.0e4,
    valid_tokens: Optional[Tensor] = None,
) -> GateResult:
    """Convert token-location probabilities into a safe hard additive attention mask.

    The forward pass exactly uses the hard threshold specified in the paper. The backward pass uses the
    sigmoid probabilities (straight-through estimator). If a threshold masks every token at a spatial
    location, the highest-probability valid token is retained to prevent an all ``-inf`` softmax row.
    """
    if probabilities.ndim != 3:
        raise ValueError(f"Expected [B, N, T] probabilities, got {tuple(probabilities.shape)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if negative_value >= 0:
        raise ValueError("negative_value must be negative to suppress attention logits")

    batch, locations, tokens = probabilities.shape
    valid = _normalise_valid_tokens(valid_tokens, batch, tokens, probabilities.device)
    soft = probabilities * valid[:, None, :].to(probabilities.dtype)
    hard = (soft > threshold).to(probabilities.dtype)

    # A masked-all row yields NaNs in softmax/SDPA. Preserve one maximally likely valid token.
    empty_rows = hard.sum(dim=-1, keepdim=True) == 0
    fallback_scores = soft.masked_fill(~valid[:, None, :], float("-inf"))
    fallback_index = fallback_scores.argmax(dim=-1, keepdim=True)
    fallback = torch.zeros_like(hard).scatter_(-1, fallback_index, 1.0)
    hard = torch.where(empty_rows, fallback, hard)

    # Exact hard values in forward; soft probabilities provide gradients in backward.
    ste = hard.detach() - soft.detach() + soft
    additive_mask = (1.0 - ste) * negative_value
    return GateResult(probabilities=probabilities, hard_gate=hard, ste_gate=ste, additive_mask=additive_mask)


class MaskGateHead(nn.Module):
    """A compact token-conditioned MLP gate predictor.

    The paper permits a shallow convolutional or MLP predictor but does not publish its exact architecture.
    This implementation uses separate MLP projections of the current latent sequence and text tokens, followed
    by a scaled compatibility score. It preserves the paper's required token-conditioned spatial probability map
    while avoiding assumptions about hidden feature-map height/width that Diffusers does not expose to processors.
    """

    def __init__(self, feature_dim: int, token_dim: int, hidden_dim: int = 128):
        super().__init__()
        if min(feature_dim, token_dim, hidden_dim) <= 0:
            raise ValueError("feature_dim, token_dim, and hidden_dim must be positive")
        self.feature_dim = feature_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.feature_net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.token_net = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.logit_bias = nn.Parameter(torch.zeros(()))
        self.scale = hidden_dim**-0.5

    def forward(self, spatial_features: Tensor, token_embeddings: Tensor) -> Tensor:
        if spatial_features.ndim != 3 or token_embeddings.ndim != 3:
            raise ValueError("MaskGateHead expects spatial_features [B,N,C] and token_embeddings [B,T,D]")
        if spatial_features.shape[0] != token_embeddings.shape[0]:
            raise ValueError("Batch size differs between spatial features and token embeddings")
        if spatial_features.shape[-1] != self.feature_dim or token_embeddings.shape[-1] != self.token_dim:
            raise ValueError(
                "Gate input dimensions do not match this gate head: "
                f"{spatial_features.shape[-1]}/{token_embeddings.shape[-1]} vs {self.feature_dim}/{self.token_dim}"
            )
        features = self.feature_net(spatial_features)
        tokens = self.token_net(token_embeddings)
        logits = torch.einsum("bnh,bth->bnt", features, tokens) * self.scale + self.logit_bias
        return torch.sigmoid(logits)
