"""Training loop for frozen-SDXL / trainable-MaskAttn gate fine-tuning."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from .config import prepare_run_directory, seed_everything
from .data import ImageCaptionDataset, collate_captions, read_jsonl
from .model import (
    MaskAttnConfig,
    assert_maskattn_device_consistency,
    assert_only_maskattn_trainable,
    install_maskattn,
    load_maskattn_checkpoint,
    maskattn_open_fraction,
    maskattn_parameters,
    save_maskattn_checkpoint,
)
from .runtime import load_sdxl_pipeline, pipeline_components_frozen, resolve_model_source


def _encode_prompts(pipe, captions: list[str], device):
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=captions,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    return prompt_embeds, pooled_prompt_embeds


def _time_ids(pipe, batch_size: int, resolution: int, dtype, device):
    projection_dim = pipe.text_encoder_2.config.projection_dim
    time_ids = pipe._get_add_time_ids(
        (resolution, resolution), (0, 0), (resolution, resolution), dtype, text_encoder_projection_dim=projection_dim
    )
    return time_ids.repeat(batch_size, 1).to(device)


def _log_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _generate_validation_samples(pipe, unet, run_dir: Path, step: int, prompts: list[str], seed: int, resolution: int) -> None:
    """Generate periodic fixed-seed samples from the current unwrapped trainable gate heads."""
    import torch

    if not prompts:
        return
    samples_dir = run_dir / "samples" / f"step_{step:07d}"
    samples_dir.mkdir(parents=True, exist_ok=True)
    was_training = unet.training
    unet.eval()
    with torch.no_grad():
        for index, prompt in enumerate(prompts):
            generator = torch.Generator(device=pipe._execution_device).manual_seed(seed + index)
            image = pipe(
                prompt, height=resolution, width=resolution, num_inference_steps=30, guidance_scale=7.5, generator=generator
            ).images[0]
            image.save(samples_dir / f"{index:02d}.png")
    unet.train(was_training)


def run_training(config: dict[str, Any], *, root: str | Path) -> Path:
    """Train the gates on the standard noise-prediction objective with every SDXL component frozen.

    `train.init_from` starts a new phase from existing gate weights, as the 1024px phase does after the 512px
    phase: the step counter, optimizer, and learning-rate schedule all start over. `train.resume_from` continues
    an interrupted run of the same phase and restores all three.
    """
    final_path = _train_gates(config, root)
    # Ablation scripts train several variants in one process, so release this run's GPU memory.
    gc.collect()
    try:
        import torch
    except ImportError:
        return final_path
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_path


def _train_gates(config: dict[str, Any], root: str | Path) -> Path:
    try:
        import torch
        from accelerate import Accelerator
        from diffusers import DDPMScheduler
        from diffusers.optimization import get_scheduler
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "Training dependencies are missing. Install the public dependencies with `pip install -e .`."
        ) from exc

    train_cfg = config["train"]
    max_train_steps = int(train_cfg["max_train_steps"])
    resolution = int(train_cfg["resolution"])
    init_path = train_cfg.get("init_from")
    resume_path = train_cfg.get("resume_from")
    if init_path and resume_path:
        raise ValueError("Set either train.init_from (new phase) or train.resume_from (same phase), not both.")

    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        mixed_precision=None if train_cfg.get("mixed_precision", "no") == "no" else train_cfg["mixed_precision"],
    )
    seed_everything(int(train_cfg["seed"]) + accelerator.process_index)
    run_dir = prepare_run_directory(config["output_dir"], config, root)
    records = read_jsonl(config["data"]["caption_cache"])
    dataset = ImageCaptionDataset(records, resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg["train_batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collate_captions,
        pin_memory=accelerator.device.type == "cuda",
    )

    model_cfg = config["model"]
    allow_download = bool(model_cfg.get("allow_download", False))
    pipe = load_sdxl_pipeline(
        model_cfg["path"],
        device=str(accelerator.device),
        dtype=model_cfg.get("dtype", "auto"),
        local_files_only=True,
        allow_download=allow_download,
    )
    pipeline_components_frozen(pipe)
    # SDXL's stock VAE is numerically unstable in fp16; Diffusers' own SDXL trainer keeps it in fp32 for this reason.
    pipe.vae.to(dtype=torch.float32)
    installation = install_maskattn(pipe.unet, MaskAttnConfig.from_mapping(config["maskattn"]))
    assert_only_maskattn_trainable(pipe.unet)
    assert_maskattn_device_consistency(pipe.unet)
    if train_cfg.get("gradient_checkpointing", False):
        # Diffusers only checkpoints in train mode. SDXL's dropout is zero, so this changes memory use only.
        pipe.unet.enable_gradient_checkpointing()
        pipe.unet.train()
    scheduler_source, is_local = resolve_model_source(model_cfg["path"], allow_download=allow_download)
    noise_scheduler = DDPMScheduler.from_pretrained(
        scheduler_source,
        subfolder="scheduler",
        local_files_only=is_local,
    )
    optimizer = torch.optim.AdamW(
        maskattn_parameters(pipe.unet),
        lr=float(train_cfg["learning_rate"]),
        betas=tuple(float(value) for value in train_cfg["betas"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(train_cfg["lr_warmup_steps"]) * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(pipe.unet, optimizer, dataloader, lr_scheduler)
    gate_unet = accelerator.unwrap_model(unet)
    assert_maskattn_device_consistency(gate_unet)
    gate_parameters = maskattn_parameters(gate_unet)

    global_step = 0
    if init_path:
        load_maskattn_checkpoint(gate_unet, init_path)
    if resume_path:
        checkpoint = load_maskattn_checkpoint(gate_unet, resume_path)
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        global_step = int(checkpoint.get("global_step") or 0)
        if global_step >= max_train_steps:
            raise ValueError(
                f"{resume_path} is already at step {global_step} of {max_train_steps}, so nothing would be trained. "
                "Use train.init_from to start a new phase from these gates."
            )

    def save(path: Path) -> None:
        save_maskattn_checkpoint(
            gate_unet,
            path,
            installation=installation,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=lr_scheduler.state_dict(),
            global_step=global_step,
        )

    progress = tqdm(total=max_train_steps, initial=global_step, disable=not accelerator.is_local_main_process, desc="MaskAttn")
    log_path = run_dir / "train_metrics.jsonl"
    data_iter = iter(dataloader)
    while global_step < max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        with accelerator.accumulate(unet):
            pixels = batch["pixel_values"].to(accelerator.device, dtype=pipe.vae.dtype)
            with torch.no_grad():
                latents = pipe.vae.encode(pixels).latent_dist.sample() * pipe.vae.config.scaling_factor
                prompt_embeds, pooled_embeds = _encode_prompts(pipe, batch["captions"], accelerator.device)
            latents = latents.to(dtype=prompt_embeds.dtype)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            added_cond_kwargs = {
                "text_embeds": pooled_embeds,
                "time_ids": _time_ids(pipe, latents.shape[0], resolution, prompt_embeds.dtype, latents.device),
            }
            with accelerator.autocast():
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs
                ).sample
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss {loss.item()} after {global_step} optimizer steps")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(gate_parameters, float(train_cfg["max_grad_norm"]))
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            global_step += 1
            progress.update(1)
            if accelerator.is_main_process and global_step % int(train_cfg["log_every"]) == 0:
                _log_jsonl(
                    log_path,
                    {
                        "step": global_step,
                        "loss": float(loss.detach().cpu()),
                        "lr": float(lr_scheduler.get_last_lr()[0]),
                        "gate_open_fraction": maskattn_open_fraction(gate_unet),
                    },
                )
            if accelerator.is_main_process and global_step % int(train_cfg["checkpointing_steps"]) == 0:
                save(run_dir / "checkpoints" / f"gate_step_{global_step:07d}.pt")
            validation_every = int(train_cfg.get("validation_every", 0))
            if validation_every and global_step % validation_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _generate_validation_samples(
                        pipe,
                        gate_unet,
                        run_dir,
                        global_step,
                        list(train_cfg.get("validation_prompts", [])),
                        int(train_cfg["seed"]),
                        resolution,
                    )
                accelerator.wait_for_everyone()
    progress.close()
    accelerator.wait_for_everyone()
    final_path = run_dir / "checkpoints" / "gate_final.pt"
    if accelerator.is_main_process:
        save(final_path)
    accelerator.free_memory()
    return final_path
