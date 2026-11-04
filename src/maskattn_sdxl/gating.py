"""Token-conditioned gate heads (Eq. 1), hard gates (Eq. 2-3), and masked attention (Eq. 4)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

MIN_PROBABILITY = 1.0e-12


@dataclass
class GateResult:
    probabilities: Tensor  # G-hat, [B, N, T]
    hard_gate: Tensor  # G, [B, N, T], exact 0/1
    additive_mask: Tensor  # M, [B, N, T], exact 0 or negative_value; carries no gradient
    soft_bias: Tensor  # log G-hat, [B, N, T]; the differentiable relaxation of M


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
    """Threshold probabilities into the hard gate G and the additive mask M of Eq. (2)-(3).

    `additive_mask` is what the forward pass applies. `soft_bias` is its continuous relaxation, used by
    `masked_attention` for the straight-through gradient. If a location would lose every token, its most
    probable valid token is kept: with -inf that row would be NaN, and with a finite constant the softmax would
    cancel the constant and leave the location unmasked.
    """
    if probabilities.ndim != 3:
        raise ValueError(f"Expected [B, N, T] probabilities, got {tuple(probabilities.shape)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if negative_value >= 0:
        raise ValueError("negative_value must be negative to suppress attention logits")

    batch, _, tokens = probabilities.shape
    valid = _normalise_valid_tokens(valid_tokens, batch, tokens, probabilities.device)
    soft = probabilities * valid[:, None, :].to(probabilities.dtype)
    hard = (soft > threshold).to(probabilities.dtype)

    empty_rows = hard.sum(dim=-1, keepdim=True) == 0
    fallback_scores = soft.detach().masked_fill(~valid[:, None, :], float("-inf"))
    fallback_index = fallback_scores.argmax(dim=-1, keepdim=True)
    fallback = torch.zeros_like(hard).scatter_(-1, fallback_index, 1.0)
    hard = torch.where(empty_rows, fallback, hard)

    return GateResult(
        probabilities=probabilities,
        hard_gate=hard,
        additive_mask=(1.0 - hard) * negative_value,
        soft_bias=torch.log(soft.clamp_min(MIN_PROBABILITY)),
    )


def masked_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    gate: GateResult,
    *,
    attention_bias: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Eq. (4): Softmax(QK^T / sqrt(d) + M) V, for [B, H, N, d] queries and [B, H, T, d] keys and values.

    The forward value uses only the hard mask M. A token masked by a large negative logit has softmax weight
    zero and therefore zero derivative, so an estimator placed on the logits could never reopen a closed gate.
    The backward path instead uses Softmax(QK^T / sqrt(d) + log G-hat), the continuous relaxation of Eq. (3):
    log 1 = 0 and log 0 = -inf.

    The attention is written out because cuDNN's scaled_dot_product_attention cannot backpropagate into an
    `attn_mask` ("LSE is not correctly aligned"). Training and inference share this one path.
    """
    logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    if attention_bias is not None:
        logits = logits + attention_bias
    weights = torch.softmax(logits + gate.additive_mask[:, None].to(logits.dtype), dim=-1)
    if gate.soft_bias.requires_grad:
        relaxed = torch.softmax(logits + gate.soft_bias[:, None].to(logits.dtype), dim=-1)
        # The bracket is exactly zero, so the forward value stays the hard attention in any precision.
        weights = weights.detach() + (relaxed - relaxed.detach())
    return torch.matmul(weights, value), weights


class MaskGateHead(nn.Module):
    """Compact token-conditioned MLP gate predictor, f(X, e_t) in Eq. (1).

    The paper allows a shallow convolutional or MLP head and does not publish its architecture. Each location
    and each token is projected separately and scored by a scaled dot product. The head starts open everywhere:
    a positive bias with a near-zero compatibility term makes an untrained model behave as plain SDXL.
    """

    def __init__(self, feature_dim: int, token_dim: int, hidden_dim: int = 192, init_logit: float = 2.0):
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
        self.logit_bias = nn.Parameter(torch.tensor(float(init_logit)))
        self.scale = hidden_dim**-0.5
        # Shrink one branch only: zeroing both would be a saddle point with no gradient on either.
        nn.init.normal_(self.feature_net[-1].weight, std=1.0e-3)
        nn.init.zeros_(self.feature_net[-1].bias)

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
