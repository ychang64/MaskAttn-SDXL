# Reproducibility notes

## Method configuration

- The SDXL U-Net backbone, VAE, text encoders, and sampling procedure are frozen. Masking heads are the trainable parameters.
- The gate head receives latent features and token embeddings, applies sigmoid, uses a 0.5 hard threshold, and uses a straight-through estimator.
- The hard gate becomes additive attention bias: `0` for active connections and `-10000.0` for suppressed connections.
- The mask is added to cross-attention logits before softmax.
- The default configuration uses mid-resolution blocks with encoder+decoder placement.
- Gate heads are shared by stage and feature dimensions by default. Set `share_gate_heads_by_stage: false` for independent heads per selected processor.
- The implementation maps high to `down_blocks.1`/`up_blocks.1`, mid to `down_blocks.2`/`up_blocks.0`, and low to `mid_block`.

## Training configuration

- COCO train2014 captions are filtered for at least two noun phrases.
- The 512px configuration uses 100k steps, effective batch size 16, AdamW (`1e-4`, weight decay `0.01`, betas `0.9/0.999`), 1k warmup, cosine decay, gradient clipping `1.0`, and mixed precision.
- The 1024px continuation uses 10k steps and batch size 8.

## Metrics and artifacts

- Quality evaluation provides FID, CLIP Score, Precision, and Recall.
- Compositional evaluation uses the official T2I-CompBench++ / UniDet and GenEval adapters.
- Every run records its resolved configuration, environment metadata, seed, and output location.
- Checkpoints contain gate weights, optimizer state, global step, selected layers, and MaskAttn configuration.
- Qualitative generation stores token-wise gate probabilities, hard masks, and post-mask attention maps.
- MaskAttn inference reconstructs its processor configuration from checkpoint metadata, records checkpoint SHA-256, and verifies processor invocation on the final pipeline U-Net.
- Random gates are restricted to the explicit `--allow-untrained-gates` integration smoke; inference, evaluation, and efficiency benchmarking require a trained gate checkpoint.

## Reference environment

| Item | Version / source |
|---|---|
| SDXL checkpoint | `stabilityai/stable-diffusion-xl-base-1.0`, revision `462165984030d82259a11f4367a4eed129e94a7b` |
| Diffusers | `v0.35.1`, commit `0f252be0ed42006c125ef4429156cb13ae6c1d60` |
| PyTorch | `>=2.4,<2.6` |
| Refiner | not used |

## Test coverage

- Gate dimensions, probability range, hard-gate behavior, and STE gradients.
- Additive mask behavior before softmax and all-masked-row stability.
- Frozen base parameters and trainable masking-head checks.
- Stage and placement selection.
- Gate checkpoint save/load.
- Real Diffusers attention-processor parity with an all-open gate.
- Local SDXL U-Net and pipeline smoke checks.
- Final-U-Net attachment, checkpoint metadata compatibility, gate device/dtype consistency, processor call counting, and checkpoint-required inference checks.
