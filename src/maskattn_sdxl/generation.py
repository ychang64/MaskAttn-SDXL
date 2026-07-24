"""Deterministic image generation and qualitative MaskAttn artifact export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import prepare_run_directory, seed_everything
from .model import (
    MaskAttnConfig,
    collect_recorded_masks,
    install_maskattn,
    load_maskattn_checkpoint,
    set_mask_recording,
)
from .runtime import load_sdxl_pipeline
from .visualize import save_token_masks


def read_prompt_records(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Prompt file not found: {source}")
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prompt JSON at {source}:{line_no}") from exc
            if "prompt" not in value:
                raise ValueError(f"Prompt record at {source}:{line_no} has no `prompt` field")
            records.append(value)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No prompt records found in {source}")
    return records


def _load_pipeline_for_generation(config: dict[str, Any]):
    pipe = load_sdxl_pipeline(
        config["model"]["path"],
        device=config["model"].get("device", "auto"),
        dtype=config["model"].get("dtype", "auto"),
        local_files_only=True,
        allow_download=bool(config["model"].get("allow_download", False)),
        disable_safety_checker=bool(config.get("disable_safety_checker", False)),
    )
    method = config.get("method", "maskattn")
    if method == "sdxl":
        return pipe, None
    if method != "maskattn":
        raise ValueError(
            f"This generator implements SDXL and MaskAttn-SDXL only, not `{method}`. "
            "Use the corresponding baseline adapter/config for other methods."
        )
    installation = install_maskattn(pipe.unet, MaskAttnConfig.from_mapping(config["maskattn"]))
    checkpoint = config.get("checkpoint")
    if checkpoint:
        load_maskattn_checkpoint(pipe.unet, checkpoint)
    return pipe, installation


def generate_records(config: dict[str, Any], records: Iterable[dict[str, Any]], *, root: str | Path) -> Path:
    """Generate one deterministic image per prompt and write portable metadata JSONL."""
    import torch

    seed_everything(int(config.get("seed", 42)))
    run_dir = prepare_run_directory(config["output_dir"], config, root)
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    pipe, installation = _load_pipeline_for_generation(config)
    device = str(pipe._execution_device)
    metadata_path = run_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata:
        for index, record in enumerate(records):
            prompt = record["prompt"]
            image_seed = int(record.get("seed", int(config.get("seed", 42)) + index))
            generator = torch.Generator(device=device).manual_seed(image_seed)
            save_masks = bool(config.get("save_masks", False)) and installation is not None
            if save_masks:
                set_mask_recording(pipe.unet, True)
            result = pipe(
                prompt=prompt,
                negative_prompt=record.get("negative_prompt", config.get("negative_prompt")),
                num_inference_steps=int(record.get("num_inference_steps", config.get("num_inference_steps", 30))),
                guidance_scale=float(record.get("guidance_scale", config.get("guidance_scale", 7.5))),
                height=int(record.get("height", config.get("height", 1024))),
                width=int(record.get("width", config.get("width", 1024))),
                generator=generator,
            )
            filename = f"{index:05d}.png"
            result.images[0].save(image_dir / filename)
            mask_dir = None
            if save_masks:
                token_labels = pipe.tokenizer.tokenize(prompt)
                mask_dir = save_token_masks(collect_recorded_masks(pipe.unet), run_dir / "masks", sample_name=f"{index:05d}", token_labels=token_labels)
                set_mask_recording(pipe.unet, False)
            output = {
                "index": index,
                "image": str((image_dir / filename).resolve()),
                "prompt": prompt,
                "negative_prompt": record.get("negative_prompt", config.get("negative_prompt")),
                "seed": image_seed,
                "method": config.get("method", "maskattn"),
                "mask_artifact": str(mask_dir.resolve()) if mask_dir else None,
            }
            metadata.write(json.dumps(output, ensure_ascii=False) + "\n")
    return run_dir


def make_comparison_grid(image_paths: list[str | Path], labels: list[str], output_path: str | Path) -> Path:
    """Create a side-by-side qualitative grid with matching seeds/images supplied by callers."""
    from PIL import Image, ImageDraw

    if len(image_paths) != len(labels) or not image_paths:
        raise ValueError("image_paths and labels must be non-empty and have the same length")
    images = [Image.open(path).convert("RGB") for path in image_paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    label_height = 28
    canvas = Image.new("RGB", (width * len(images), height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(image.resize((width, height)), (index * width, label_height))
        draw.text((index * width + 4, 6), label, fill="black")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return target
