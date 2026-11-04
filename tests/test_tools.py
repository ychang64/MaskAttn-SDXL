"""Experiment tooling: baselines, orchestration, data preparation, visualisation, and adapters."""

from __future__ import annotations

import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _baseline_config(name: str) -> dict:
    from generate_baseline import DEFAULTS

    with (ROOT / "configs" / "baselines.yaml").open(encoding="utf-8") as handle:
        return {**DEFAULTS, **yaml.safe_load(handle), "baseline": name}


def test_baseline_name_selects_its_own_model_and_output() -> None:
    from generate_baseline import resolve_baseline

    resolved = resolve_baseline(_baseline_config("sd15"))
    assert resolved["model_path"] == "models/sd15"
    assert resolved["output_dir"] == "outputs/baseline_sd15"
    assert resolved["resolution"] == 512


def test_external_and_unknown_baselines_are_rejected() -> None:
    from generate_baseline import resolve_baseline

    with pytest.raises(ValueError, match="official repository"):
        resolve_baseline(_baseline_config("attend_and_excite"))
    with pytest.raises(ValueError, match="Unknown baseline"):
        resolve_baseline(_baseline_config("missing"))


def test_run_all_trains_both_phases_and_passes_the_final_gates_on() -> None:
    from run_all import DEFAULTS, build_command

    assert DEFAULTS["run"][:2] == ["train_512", "train_1024"]
    assert any("train_1024.yaml" in part for part in build_command("train_1024", DEFAULTS))
    assert f"checkpoint={DEFAULTS['checkpoint']}" in build_command("quality", DEFAULTS)


def test_efficiency_benchmark_holds_one_pipeline_at_a_time(monkeypatch, tmp_path) -> None:
    import benchmark_efficiency

    class FakePipe:
        pass

    alive: list[weakref.ref] = []

    def fake_load(config):
        assert all(reference() is None for reference in alive)
        pipe = FakePipe()
        alive.append(weakref.ref(pipe))
        return pipe, None, {}

    monkeypatch.setattr("maskattn_sdxl.generation._load_pipeline_for_generation", fake_load)
    monkeypatch.setattr("maskattn_sdxl.benchmarks.benchmark_pipeline", lambda pipe, **kwargs: {"latency_seconds_per_image": 1.0})
    monkeypatch.setattr(sys, "argv", ["benchmark_efficiency.py", "--set", "checkpoint=gates.pt", "--output-dir", str(tmp_path)])
    benchmark_efficiency.main()
    assert len(alive) == 2


def test_coco_limit_is_a_seeded_random_subset(monkeypatch, tmp_path) -> None:
    from maskattn_sdxl import data

    class FakeNlp:
        def pipe(self, captions, batch_size):
            for caption in captions:
                yield SimpleNamespace(text=caption, noun_chunks=[SimpleNamespace(text="a"), SimpleNamespace(text="b")])

    monkeypatch.setattr(data, "_require_spacy", lambda name: FakeNlp())
    images = tmp_path / "images"
    images.mkdir()
    annotations = tmp_path / "captions.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": index, "file_name": f"{index}.jpg"} for index in range(100)],
                "annotations": [{"image_id": index, "caption": f"caption {index}"} for index in range(100)],
            }
        ),
        encoding="utf-8",
    )

    def sample(seed: int) -> list[int]:
        output = tmp_path / f"cache_{seed}.jsonl"
        assert data.build_coco_caption_cache(annotations, images, output, limit=10, seed=seed) == 10
        return [record["image_id"] for record in data.read_jsonl(output)]

    assert sample(1) == sample(1)
    assert sample(1) != sample(2)
    assert sample(1) != list(range(10))


def test_flickr30k_eval_split_pairs_prompts_with_real_images(tmp_path) -> None:
    from PIL import Image

    from maskattn_sdxl.data import build_flickr30k_eval_split, read_jsonl

    images = tmp_path / "images"
    images.mkdir()
    lines = []
    for index in range(5):
        Image.new("RGB", (8, 8), (index, 0, 0)).save(images / f"{index}.jpg")
        lines += [f"{index}.jpg#0\tfirst caption of {index}", f"{index}.jpg#1\tsecond caption of {index}"]
    token_file = tmp_path / "results.token"
    token_file.write_text("\n".join(lines), encoding="utf-8")

    written = build_flickr30k_eval_split(token_file, images, tmp_path / "prompts.jsonl", tmp_path / "real", count=3)
    records = read_jsonl(tmp_path / "prompts.jsonl")
    assert written == len(records) == 3
    assert sorted(path.name for path in (tmp_path / "real").iterdir()) == [record["image"] for record in records]


def test_mask_png_comes_from_the_conditional_branch(tmp_path) -> None:
    from PIL import Image

    from maskattn_sdxl.visualize import save_token_masks

    unconditional_then_conditional = torch.stack([torch.zeros(4, 3), torch.ones(4, 3)])
    masks = {"layer": {"probabilities": unconditional_then_conditional, "hard_gate": unconditional_then_conditional}}
    root = save_token_masks(masks, tmp_path, sample_name="s", token_labels=["<|startoftext|>", "cat"])
    rendered = sorted(path.name for path in (root / "layer" / "prob").iterdir())
    assert len(rendered) == 2
    with Image.open(root / "layer" / "prob" / rendered[1]) as image:
        assert image.getextrema() == (255, 255)


def test_external_command_keeps_windows_paths(monkeypatch) -> None:
    from maskattn_sdxl import benchmarks

    monkeypatch.setattr(benchmarks, "os", SimpleNamespace(name="nt"))
    assert benchmarks._split_command(r'python W:\tools\eval.py --images "{images}"') == [
        "python",
        r"W:\tools\eval.py",
        "--images",
        "{images}",
    ]
