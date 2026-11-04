"""End-to-end training and generation on an offline tiny SDXL pipeline."""

from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("torchvision")

from tiny_sdxl import build_tiny_sdxl, write_tiny_dataset

from maskattn_sdxl.generation import generate_records, prompt_token_labels
from maskattn_sdxl.model import read_maskattn_checkpoint
from maskattn_sdxl.training import run_training


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    return build_tiny_sdxl(root / "model"), write_tiny_dataset(root / "data", count=8, resolution=64)


def _config(assets, output_dir, **train_overrides) -> dict:
    model, cache = assets
    train = {
        "resolution": 64,
        "max_train_steps": 3,
        "train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "lr_warmup_steps": 1,
        "mixed_precision": "no",
        "max_grad_norm": 1.0,
        "seed": 42,
        "checkpointing_steps": 1000,
        "log_every": 1,
        "validation_every": 0,
        "validation_prompts": [],
    }
    train.update(train_overrides)
    return {
        "model": {"path": str(model), "device": "auto", "dtype": "float32", "allow_download": False},
        "data": {"caption_cache": str(cache), "num_workers": 0},
        "maskattn": {"stage": "mid", "placement": "encoder_decoder", "gate_hidden_dim": 8},
        "train": train,
        "output_dir": str(output_dir),
    }


def _logged(run_dir) -> list[dict]:
    with (run_dir / "train_metrics.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _gate_values(checkpoint) -> list[torch.Tensor]:
    state = read_maskattn_checkpoint(checkpoint)["gate_state_dict"]
    return [value for head in state.values() for value in head.values()]


def test_training_changes_gates_and_starts_from_open_gates(assets, tmp_path) -> None:
    final = run_training(_config(assets, tmp_path / "run"), root=tmp_path)
    payload = read_maskattn_checkpoint(final)
    assert payload["global_step"] == 3
    assert payload["checkpoint_kind"] == "trained"
    log = _logged(tmp_path / "run")
    assert [row["step"] for row in log] == [1, 2, 3]
    assert log[0]["gate_open_fraction"] == 1.0


def test_new_phase_trains_from_the_previous_gates(assets, tmp_path) -> None:
    """The 1024px phase has fewer steps than the 512px phase it continues, and must still train."""
    first = run_training(_config(assets, tmp_path / "phase1", max_train_steps=4), root=tmp_path)
    second = run_training(_config(assets, tmp_path / "phase2", max_train_steps=3, init_from=str(first)), root=tmp_path)
    assert read_maskattn_checkpoint(second)["global_step"] == 3
    assert [row["step"] for row in _logged(tmp_path / "phase2")] == [1, 2, 3]
    assert not all(torch.equal(a, b) for a, b in zip(_gate_values(first), _gate_values(second)))


def test_resume_restores_step_and_schedule(assets, tmp_path) -> None:
    run_training(_config(assets, tmp_path / "run", max_train_steps=2, checkpointing_steps=2), root=tmp_path)
    saved = tmp_path / "run" / "checkpoints" / "gate_step_0000002.pt"
    assert read_maskattn_checkpoint(saved)["lr_scheduler"]["last_epoch"] == 2
    final = run_training(_config(assets, tmp_path / "run2", max_train_steps=4, resume_from=str(saved)), root=tmp_path)
    assert read_maskattn_checkpoint(final)["global_step"] == 4
    assert [row["step"] for row in _logged(tmp_path / "run2")] == [3, 4]


def test_resume_past_the_end_fails_instead_of_skipping_training(assets, tmp_path) -> None:
    first = run_training(_config(assets, tmp_path / "phase1", max_train_steps=4), root=tmp_path)
    with pytest.raises(ValueError, match="init_from"):
        run_training(_config(assets, tmp_path / "phase2", max_train_steps=3, resume_from=str(first)), root=tmp_path)


def test_init_from_and_resume_from_are_exclusive(assets, tmp_path) -> None:
    with pytest.raises(ValueError, match="not both"):
        run_training(_config(assets, tmp_path / "run", init_from="a.pt", resume_from="b.pt"), root=tmp_path)


def test_gradient_checkpointing_trains(assets, tmp_path) -> None:
    final = run_training(_config(assets, tmp_path / "run", max_train_steps=1, gradient_checkpointing=True), root=tmp_path)
    assert read_maskattn_checkpoint(final)["global_step"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 mixed precision needs a GPU")
def test_fp16_mixed_precision_trains(assets, tmp_path) -> None:
    from accelerate.state import AcceleratorState

    config = _config(
        assets,
        tmp_path / "run",
        mixed_precision="fp16",
        gradient_checkpointing=True,
        validation_every=1,
        validation_prompts=["a red cat"],
    )
    config["model"]["dtype"] = "float16"
    # Accelerate fixes the precision once per process, and earlier tests already chose "no".
    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        final = run_training(config, root=tmp_path)
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
    assert read_maskattn_checkpoint(final)["global_step"] == 3
    assert all(torch.isfinite(value).all() for value in _gate_values(final))


def test_generation_exports_conditional_masks_with_aligned_labels(assets, tmp_path) -> None:
    checkpoint = run_training(_config(assets, tmp_path / "train", max_train_steps=1), root=tmp_path)
    config = {
        "method": "maskattn",
        "checkpoint": str(checkpoint),
        "model": {"path": str(assets[0]), "device": "cpu", "dtype": "float32"},
        "output_dir": str(tmp_path / "generated"),
        "seed": 1,
        "height": 64,
        "width": 64,
        "num_inference_steps": 2,
        "guidance_scale": 7.5,
        "save_masks": True,
    }
    run_dir = generate_records(config, [{"prompt": "a cat"}], root=tmp_path)
    metadata = json.loads((run_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metadata["runtime_audit"]["maskattn_forward_calls"] > 0
    layer_dirs = [path for path in (run_dir / "masks" / "00000").iterdir() if path.is_dir()]
    names = sorted(path.name for path in (layer_dirs[0] / "prob").iterdir())
    assert names[0].startswith("000___startoftext")
    assert names[-1].split("_", 1)[1].strip("_").startswith("endoftext")


def test_prompt_token_labels_start_with_bos(assets) -> None:
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(str(assets[0] / "tokenizer"))
    labels = prompt_token_labels(tokenizer, "a cat")
    assert labels[0] == "<|startoftext|>"
    assert labels[-1] == "<|endoftext|>"
    assert len(labels) == len(tokenizer("a cat").input_ids)
