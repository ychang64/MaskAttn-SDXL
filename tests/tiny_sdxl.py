"""Offline, randomly initialised SDXL pipeline with SDXL's block layout. Needs no weights and no network."""

from __future__ import annotations

import json
from pathlib import Path

SDXL_UNET_CONFIG = {
    "sample_size": 128,
    "in_channels": 4,
    "out_channels": 4,
    "layers_per_block": 2,
    "block_out_channels": (320, 640, 1280),
    "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    "up_block_types": ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "transformer_layers_per_block": (1, 2, 10),
    "attention_head_dim": (5, 10, 20),
    "cross_attention_dim": 2048,
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 256,
    "projection_class_embeddings_input_dim": 2816,
    "use_linear_projection": True,
}


def build_sdxl_unet():
    """The real SDXL Base U-Net architecture (2.57B parameters) with random weights."""
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel(**SDXL_UNET_CONFIG)


def _write_tokenizer(directory: Path, pad_token: str):
    from transformers import CLIPTokenizer

    letters = [chr(code) for code in range(ord("a"), ord("z") + 1)]
    vocab = letters + [letter + "</w>" for letter in letters] + ["!</w>", "<|startoftext|>", "<|endoftext|>"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "vocab.json").write_text(json.dumps({token: index for index, token in enumerate(vocab)}), encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    return CLIPTokenizer(str(directory / "vocab.json"), str(directory / "merges.txt"), model_max_length=77, pad_token=pad_token)


def build_tiny_sdxl(directory: str | Path) -> Path:
    """Save a tiny pipeline in the on-disk layout `load_sdxl_pipeline` expects, fp16 variant included."""
    import torch
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    root = Path(directory)
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16, 32),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        mid_block_type="UNetMidBlock2DCrossAttn",
        transformer_layers_per_block=(1, 1, 2),
        attention_head_dim=(1, 2, 4),
        norm_num_groups=4,
        cross_attention_dim=32,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=68,
        use_linear_projection=True,
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(4, 4, 4, 4),
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        norm_num_groups=2,
        sample_size=32,
        force_upcast=True,
    )
    tokenizer = _write_tokenizer(root / "_tokenizer_1", "<|endoftext|>")
    tokenizer_2 = _write_tokenizer(root / "_tokenizer_2", "!</w>")
    common = dict(vocab_size=len(tokenizer), intermediate_size=16, num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=77)
    text_encoder = CLIPTextModel(CLIPTextConfig(hidden_size=12, projection_dim=12, **common))
    text_encoder_2 = CLIPTextModelWithProjection(CLIPTextConfig(hidden_size=20, projection_dim=20, **common))
    scheduler = EulerDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1,
        timestep_spacing="leading",
        prediction_type="epsilon",
    )
    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler,
    )
    target = root / "tiny-sdxl"
    pipe.save_pretrained(target, variant="fp16", safe_serialization=True)
    return target


def write_tiny_dataset(directory: str | Path, *, count: int = 8, resolution: int = 32) -> Path:
    """Write solid-colour images and a caption cache in the JSONL format the trainer reads."""
    from PIL import Image

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "captions.jsonl"
    with cache.open("w", encoding="utf-8") as handle:
        for index in range(count):
            image_path = root / f"{index:03d}.png"
            Image.new("RGB", (resolution, resolution), (index * 30 % 255, 40, 200)).save(image_path)
            record = {"image_id": index, "image_path": str(image_path), "caption": "a red cat on a blue car", "noun_phrases": ["a red cat", "a blue car"]}
            handle.write(json.dumps(record) + "\n")
    return cache
