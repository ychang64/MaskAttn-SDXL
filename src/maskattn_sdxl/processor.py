"""A Diffusers-compatible cross-attention processor with pre-softmax MaskAttn bias."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .gating import GateResult, MaskGateHead, hard_gate_with_ste


def _valid_tokens_from_attention_mask(
    attention_mask: Optional[Tensor], batch: int, tokens: int, device: torch.device
) -> Optional[Tensor]:
    """Best-effort validity extraction for the fallback-token safeguard.

    Diffusers normally passes no encoder mask for SDXL. When one is supplied, it may be a binary keep-mask or an
    additive bias; both common forms are handled here. The original mask is still added to attention separately.
    """
    if attention_mask is None:
        return None
    mask = attention_mask
    while mask.ndim > 2:
        mask = mask[:, 0]
    if mask.ndim != 2 or mask.shape[-1] != tokens:
        return None
    if mask.shape[0] != batch:
        return None
    if mask.dtype == torch.bool:
        return mask.to(device)
    if torch.is_floating_point(mask) and float(mask.min().detach().cpu()) < -1.0:
        return (mask > -1.0e3).to(device)
    return (mask > 0).to(device)


class MaskAttnProcessor(nn.Module):
    """Drop-in processor for Diffusers ``Attention`` modules.

    It mirrors Diffusers' ``AttnProcessor2_0`` projections and output path, but calls scaled-dot-product attention
    with an additive gate bias of shape ``[B, 1, N, T]``. The gate is therefore applied to QK logits before softmax.
    It intentionally targets cross-attention only; installation rejects self-attention processors.
    """

    def __init__(
        self,
        gate_head: MaskGateHead,
        *,
        threshold: float = 0.5,
        negative_value: float = -1.0e4,
        layer_name: str = "",
    ):
        super().__init__()
        self.gate_head = gate_head
        self.threshold = float(threshold)
        self.negative_value = float(negative_value)
        self.layer_name = layer_name
        self.record_masks = False
        self.record_attention = True
        self.last_gate: Optional[dict[str, Tensor]] = None

    def set_record_masks(self, enabled: bool, *, include_attention: bool = True) -> None:
        self.record_masks = bool(enabled)
        self.record_attention = bool(include_attention)
        if not enabled:
            self.last_gate = None

    def _record(self, result: GateResult) -> None:
        if not self.record_masks:
            return
        # CPU copies intentionally sever the graph and keep qualitative visualisation bounded to an explicit mode.
        self.last_gate = {
            "probabilities": result.probabilities.detach().float().cpu(),
            "hard_gate": result.hard_gate.detach().float().cpu(),
        }

    def forward(
        self,
        attn,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        temb: Optional[Tensor] = None,
        **_: object,
    ) -> Tensor:
        if encoder_hidden_states is None:
            raise ValueError("MaskAttnProcessor can only be installed on cross-attention (`attn2`) modules.")

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channels, height * width).transpose(1, 2)
        elif input_ndim == 3:
            batch = hidden_states.shape[0]
        else:
            raise ValueError(f"Expected 3D or 4D attention states, got {tuple(hidden_states.shape)}")

        # Keep the pre-key-projection text representation for f(X, e_t), as specified in the paper.
        gate_text = encoder_hidden_states
        token_count = encoder_hidden_states.shape[1]
        valid_tokens = _valid_tokens_from_attention_mask(attention_mask, batch, token_count, hidden_states.device)

        # Gate MLPs remain fp32 for stable sigmoid/STE arithmetic under fp16 mixed precision.
        probabilities = self.gate_head(hidden_states.float(), gate_text.float())
        gate = hard_gate_with_ste(
            probabilities,
            threshold=self.threshold,
            negative_value=self.negative_value,
            valid_tokens=valid_tokens,
        )
        self._record(gate)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Gate bias is broadcast across heads. SDPA adds it to QK^T / sqrt(d) before its softmax.
        attn_mask = gate.additive_mask[:, None].to(dtype=query.dtype)
        if attention_mask is not None:
            prepared = attn.prepare_attention_mask(attention_mask, token_count, batch)
            prepared = prepared.view(batch, attn.heads, -1, prepared.shape[-1])
            if prepared.dtype == torch.bool:
                prepared = torch.where(
                    prepared,
                    torch.zeros((), device=query.device, dtype=query.dtype),
                    torch.full((), self.negative_value, device=query.device, dtype=query.dtype),
                )
            attn_mask = attn_mask + prepared.to(dtype=query.dtype)

        if self.record_masks and self.record_attention and self.last_gate is not None:
            # SDPA does not expose probabilities. Recompute only in explicit qualitative-recording mode;
            # this does not affect the actual attention path or gradients.
            with torch.no_grad():
                scale = float(getattr(attn, "scale", head_dim**-0.5))
                scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
                scores = scores + attn_mask.float()
                self.last_gate["attention"] = torch.softmax(scores, dim=-1).mean(dim=1).cpu()

        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        hidden_states = attended.transpose(1, 2).reshape(batch, -1, attn.heads * head_dim).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channels, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
