"""MaskAttn-SDXL: token-conditioned hard spatial gates for SDXL cross-attention.

Heavy torch imports are intentionally lazy so every experiment's ``--help`` and ``--dry-run`` mode works without an
installed GPU runtime.
"""

__all__ = [
    "MaskAttnConfig",
    "MaskGateHead",
    "hard_gate_with_ste",
    "install_maskattn",
    "load_maskattn_checkpoint",
    "save_maskattn_checkpoint",
    "load_maskattn_pipeline",
]


def __getattr__(name: str):
    if name in {"MaskGateHead", "hard_gate_with_ste"}:
        from .gating import MaskGateHead, hard_gate_with_ste

        return {"MaskGateHead": MaskGateHead, "hard_gate_with_ste": hard_gate_with_ste}[name]
    if name in {"MaskAttnConfig", "install_maskattn", "load_maskattn_checkpoint", "save_maskattn_checkpoint"}:
        from .model import MaskAttnConfig, install_maskattn, load_maskattn_checkpoint, save_maskattn_checkpoint

        return {
            "MaskAttnConfig": MaskAttnConfig,
            "install_maskattn": install_maskattn,
            "load_maskattn_checkpoint": load_maskattn_checkpoint,
            "save_maskattn_checkpoint": save_maskattn_checkpoint,
        }[name]
    if name == "load_maskattn_pipeline":
        from .runtime import load_maskattn_pipeline

        return load_maskattn_pipeline
    raise AttributeError(name)
