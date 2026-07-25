"""Install, configure, save, and load MaskAttn processors without modifying Diffusers source."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch
from torch import nn

from .gating import MaskGateHead
from .processor import MaskAttnProcessor

Stage = Literal["high", "mid", "low", "all"]
Placement = Literal["encoder", "decoder", "encoder_decoder"]


@dataclass(frozen=True)
class MaskAttnConfig:
    """Implementation settings; defaults intentionally match the paper where specified."""

    stage: Stage = "mid"
    placement: Placement = "encoder_decoder"
    threshold: float = 0.5
    negative_value: float = -1.0e4
    gate_hidden_dim: int = 128
    share_gate_heads_by_stage: bool = True

    def __post_init__(self) -> None:
        if self.stage not in {"high", "mid", "low", "all"}:
            raise ValueError(f"Unknown stage: {self.stage}")
        if self.placement not in {"encoder", "decoder", "encoder_decoder"}:
            raise ValueError(f"Unknown placement: {self.placement}")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must lie in [0, 1]")
        if self.negative_value >= 0:
            raise ValueError("negative_value must be negative")

    @classmethod
    def from_mapping(cls, values: dict) -> "MaskAttnConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass
class MaskAttnInstallation:
    config: MaskAttnConfig
    selected_layers: list[str]
    stage_by_layer: dict[str, str]

    def as_dict(self) -> dict:
        return {"config": asdict(self.config), "selected_layers": self.selected_layers, "stage_by_layer": self.stage_by_layer}


def _is_cross_attention_processor_name(name: str) -> bool:
    return ".attn2.processor" in name


def _location(name: str) -> str:
    if name.startswith("down_blocks."):
        return "encoder"
    if name.startswith("up_blocks."):
        return "decoder"
    if name.startswith("mid_block."):
        return "bridge"
    return "other"


def infer_stage(name: str) -> str:
    """Map SDXL's topology to high/mid/low ablation labels.

    This is an explicit implementation assumption: the paper names resolution stages but does not publish exact
    module paths. We map the outer cross-attention blocks to high, inner encoder/decoder blocks to mid, and the
    U-Net bridge to low. Selected paths are always recorded with a run.
    """
    if name.startswith("mid_block."):
        return "low"
    if name.startswith("down_blocks.1") or name.startswith("up_blocks.1"):
        return "high"
    if name.startswith("down_blocks.2") or name.startswith("up_blocks.0"):
        return "mid"
    return "other"


def _placement_selected(name: str, placement: Placement) -> bool:
    location = _location(name)
    if placement == "encoder":
        return location == "encoder"
    if placement == "decoder":
        return location == "decoder"
    return location in {"encoder", "decoder", "bridge"}


def selected_cross_attention_names(unet, config: MaskAttnConfig) -> tuple[list[str], dict[str, str]]:
    selected: list[str] = []
    stages: dict[str, str] = {}
    for name in sorted(unet.attn_processors):
        if not _is_cross_attention_processor_name(name) or not _placement_selected(name, config.placement):
            continue
        stage = infer_stage(name)
        if stage == "other" or (config.stage != "all" and stage != config.stage):
            continue
        selected.append(name)
        stages[name] = stage
    if not selected:
        raise ValueError(
            f"No SDXL cross-attention processors matched stage={config.stage}, placement={config.placement}. "
            "Inspect the local Diffusers architecture or use a supported SDXL UNet."
        )
    return selected, stages


def _attention_module(unet, processor_name: str):
    return unet.get_submodule(processor_name.removesuffix(".processor"))


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is None:
        raise ValueError(f"Cannot infer device for parameter-free module {type(module).__name__}")
    return parameter.device


def freeze_base_parameters(unet: nn.Module) -> None:
    for parameter in unet.parameters():
        parameter.requires_grad_(False)


def install_maskattn(unet: nn.Module, config: MaskAttnConfig) -> MaskAttnInstallation:
    """Replace selected Diffusers cross-attention processors with MaskAttn processors.

    The U-Net's original layers are untouched; only its supported processor extension point is used. Base parameters
    are frozen before gate modules are attached, leaving only gate-head parameters trainable by default.
    """
    freeze_base_parameters(unet)
    selected, stages = selected_cross_attention_names(unet, config)
    processors = dict(unet.attn_processors)
    shared_heads: dict[tuple, MaskGateHead] = {}
    for name in selected:
        attention = _attention_module(unet, name)
        feature_dim = int(attention.query_dim)
        token_dim = int(attention.cross_attention_dim)
        stage_key = stages[name] if config.share_gate_heads_by_stage else name
        key = (stage_key, feature_dim, token_dim)
        gate_head = shared_heads.get(key)
        if gate_head is None:
            # Gates intentionally remain fp32 for stable sigmoid/STE arithmetic, while their device follows
            # the final U-Net attention module. This also covers installation after ``pipe.to(device)``.
            gate_head = MaskGateHead(feature_dim, token_dim, config.gate_hidden_dim).to(
                device=_module_device(attention), dtype=torch.float32
            )
            shared_heads[key] = gate_head
        processors[name] = MaskAttnProcessor(
            gate_head,
            threshold=config.threshold,
            negative_value=config.negative_value,
            layer_name=name,
        )
    unet.set_attn_processor(processors)
    installation = MaskAttnInstallation(config=config, selected_layers=selected, stage_by_layer=stages)
    unet._maskattn_installation = installation  # type: ignore[attr-defined]
    unet._maskattn_runtime = {  # type: ignore[attr-defined]
        "checkpoint": None,
        "checkpoint_hash": None,
        "checkpoint_kind": "untrained",
        "trained_checkpoint_loaded": False,
        "allow_untrained_gates": False,
    }
    assert_maskattn_installed(unet, expected_layers=selected)
    assert_maskattn_device_consistency(unet)
    return installation


def iter_maskattn_processors(unet: nn.Module) -> Iterable[tuple[str, MaskAttnProcessor]]:
    for name, processor in unet.attn_processors.items():
        if isinstance(processor, MaskAttnProcessor):
            yield name, processor


def maskattn_parameters(unet: nn.Module) -> list[nn.Parameter]:
    # ``dict`` removes shared gate parameters that are referenced by several processors.
    unique = {id(parameter): parameter for _, processor in iter_maskattn_processors(unet) for parameter in processor.parameters()}
    return list(unique.values())


def assert_only_maskattn_trainable(unet: nn.Module) -> None:
    expected = {id(parameter) for parameter in maskattn_parameters(unet)}
    actual = {id(parameter) for parameter in unet.parameters() if parameter.requires_grad}
    if actual != expected:
        raise AssertionError("Unexpected trainable U-Net parameters; only MaskAttn gate heads may require gradients.")


def assert_maskattn_installed(unet: nn.Module, *, expected_layers: Iterable[str] | None = None) -> list[str]:
    """Verify that the final U-Net owns MaskAttn processors at every selected cross-attention layer."""
    installation = getattr(unet, "_maskattn_installation", None)
    if installation is None:
        raise AssertionError("MaskAttn is not installed on this U-Net")
    selected = list(installation.selected_layers)
    if not selected:
        raise AssertionError("MaskAttn selected no cross-attention layers")
    actual = {name for name, processor in iter_maskattn_processors(unet) if isinstance(processor, MaskAttnProcessor)}
    if actual != set(selected):
        raise AssertionError(f"Installed MaskAttn processors differ from selected layers. selected={selected}, actual={sorted(actual)}")
    if expected_layers is not None and selected != list(expected_layers):
        raise AssertionError(f"MaskAttn selected layers differ from expected layers. selected={selected}, expected={list(expected_layers)}")
    return selected


def assert_maskattn_device_consistency(unet: nn.Module) -> None:
    """Require fp32 gates on the same device as the U-Net they extend."""
    unet_device = _module_device(unet)
    for name, processor in iter_maskattn_processors(unet):
        for parameter in processor.gate_head.parameters():
            if parameter.device != unet_device:
                raise AssertionError(f"Gate parameter for {name} is on {parameter.device}, expected {unet_device}")
            if parameter.dtype != torch.float32:
                raise AssertionError(f"Gate parameter for {name} has dtype {parameter.dtype}; MaskAttn gates must be fp32")


def assert_maskattn_config_matches(unet: nn.Module, expected_config: MaskAttnConfig) -> None:
    installation = getattr(unet, "_maskattn_installation", None)
    if installation is None:
        raise AssertionError("MaskAttn is not installed on this U-Net")
    if installation.config != expected_config:
        actual = installation.config
        raise AssertionError(f"MaskAttn installation config differs from expected config. actual={actual}, expected={expected_config}")


def reset_maskattn_forward_calls(unet: nn.Module) -> None:
    for _, processor in iter_maskattn_processors(unet):
        processor.reset_forward_calls()


def maskattn_forward_calls(unet: nn.Module) -> int:
    return sum(processor.forward_calls for _, processor in iter_maskattn_processors(unet))


def _checkpoint_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_mask_recording(unet: nn.Module, enabled: bool, *, include_attention: bool = True) -> None:
    for _, processor in iter_maskattn_processors(unet):
        processor.set_record_masks(enabled, include_attention=include_attention)


def collect_recorded_masks(unet: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {name: processor.last_gate for name, processor in iter_maskattn_processors(unet) if processor.last_gate is not None}


def gate_state_dict(unet: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.cpu() for key, value in unet.state_dict().items() if ".processor.gate_head." in key}


def save_maskattn_checkpoint(
    unet: nn.Module,
    output_path: str | Path,
    *,
    installation: MaskAttnInstallation,
    optimizer_state: dict | None = None,
    global_step: int | None = None,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "maskattn-sdxl-gates-v1",
            "checkpoint_kind": "trained" if global_step is not None else "test_only",
            "maskattn_config": asdict(installation.config),
            "selected_layers": installation.selected_layers,
            "stage_by_layer": installation.stage_by_layer,
            "gate_state_dict": gate_state_dict(unet),
            "optimizer": optimizer_state,
            "global_step": global_step,
        },
        target,
    )
    with target.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint_kind": "trained" if global_step is not None else "test_only",
                "maskattn": installation.as_dict(),
                "global_step": global_step,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return target


def load_maskattn_checkpoint(unet: nn.Module, checkpoint_path: str | Path, *, strict: bool = True) -> dict:
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "maskattn-sdxl-gates-v1":
        raise ValueError("Not a MaskAttn-SDXL gate checkpoint")
    checkpoint_config = MaskAttnConfig.from_mapping(checkpoint["maskattn_config"])
    assert_maskattn_config_matches(unet, checkpoint_config)
    checkpoint_layers = list(checkpoint.get("selected_layers", []))
    assert_maskattn_installed(unet, expected_layers=checkpoint_layers)
    expected_keys = set(gate_state_dict(unet))
    checkpoint_keys = set(checkpoint["gate_state_dict"])
    if expected_keys != checkpoint_keys:
        raise RuntimeError(
            "Gate checkpoint keys do not match the final installed U-Net. "
            f"missing={sorted(expected_keys - checkpoint_keys)}, unexpected={sorted(checkpoint_keys - expected_keys)}"
        )
    result = unet.load_state_dict(checkpoint["gate_state_dict"], strict=False)
    missing_gate_keys = [key for key in result.missing_keys if ".processor.gate_head." in key]
    unexpected_gate_keys = [key for key in result.unexpected_keys if ".processor.gate_head." in key]
    if strict and (missing_gate_keys or unexpected_gate_keys):
        raise RuntimeError(f"Gate checkpoint mismatch. missing={missing_gate_keys}, unexpected={unexpected_gate_keys}")
    checkpoint_kind = checkpoint.get("checkpoint_kind", "trained" if checkpoint.get("global_step") is not None else "test_only")
    if checkpoint_kind not in {"trained", "test_only"}:
        raise ValueError(f"Unsupported MaskAttn checkpoint kind: {checkpoint_kind}")
    runtime = getattr(unet, "_maskattn_runtime", {})
    runtime.update(
        {
            "checkpoint": str(checkpoint_file.resolve()),
            "checkpoint_hash": _checkpoint_hash(checkpoint_file),
            "checkpoint_kind": checkpoint_kind,
            "trained_checkpoint_loaded": checkpoint_kind == "trained",
            "allow_untrained_gates": False,
        }
    )
    unet._maskattn_runtime = runtime  # type: ignore[attr-defined]
    assert_maskattn_device_consistency(unet)
    return checkpoint


def assert_maskattn_ready(
    unet: nn.Module,
    *,
    checkpoint_required: bool,
    expected_config: MaskAttnConfig | None = None,
    require_forward_calls: bool = False,
) -> dict:
    """Fail fast unless this exact U-Net is ready to execute a declared MaskAttn run."""
    selected = assert_maskattn_installed(unet)
    if expected_config is not None:
        assert_maskattn_config_matches(unet, expected_config)
    assert_maskattn_device_consistency(unet)
    runtime = dict(getattr(unet, "_maskattn_runtime", {}))
    if checkpoint_required and not runtime.get("trained_checkpoint_loaded"):
        raise RuntimeError(
            "MaskAttn-SDXL inference requires a trained gate checkpoint. Provide `checkpoint=/path/to/gate_final.pt`."
        )
    calls = maskattn_forward_calls(unet)
    if require_forward_calls and calls <= 0:
        raise AssertionError("MaskAttn processors were installed but were not invoked by the executed forward pass")
    gate_parameters = maskattn_parameters(unet)
    return {
        "selected_layers": selected,
        "gate_parameter_count": sum(parameter.numel() for parameter in gate_parameters),
        "gate_device": str(gate_parameters[0].device),
        "gate_dtype": str(gate_parameters[0].dtype).replace("torch.", ""),
        "maskattn_forward_calls": calls,
        **runtime,
    }
