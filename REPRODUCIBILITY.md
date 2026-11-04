# Reproducibility notes

## Method configuration

- The SDXL U-Net backbone, VAE, text encoders, and sampling procedure are frozen. Masking heads are the trainable parameters.
- The gate head receives latent features and token embeddings, applies sigmoid, uses a 0.5 hard threshold, and uses a straight-through estimator.
- The hard gate becomes additive attention bias: `0` for active connections and `-10000.0` for suppressed connections.
- The mask is added to cross-attention logits before softmax. The forward pass uses this hard mask; the backward pass uses `log G-hat` in its place, so closed gates still receive gradients.
- Gates start open everywhere, so an untrained model behaves as plain SDXL.
- The default configuration uses mid-resolution blocks with encoder+decoder placement.
- Each spatial transformer owns one gate head by default (`gate_sharing: block`). With `gate_hidden_dim: 192` the default stage adds 3,600,645 parameters. The paper states the 3.6M total but not this architecture, so the head size is fitted to that figure. `gate_sharing` also accepts `stage` and `layer`.
- The implementation maps high to `down_blocks.1`/`up_blocks.1`, mid to `down_blocks.2`/`up_blocks.0`, and low to `mid_block`.

## Training configuration

- COCO train2014 captions are filtered for at least two noun phrases.
- COCO captions are sampled in seeded random order, so `--limit 200000` is a random subset.
- The 512px configuration uses 100k steps, effective batch size 16, AdamW (`1e-4`, weight decay `0.01`, betas `0.9/0.999`), 1k warmup, cosine decay, gradient clipping `1.0`, and mixed precision.
- The 1024px continuation uses 10k steps and effective batch size 8. It starts from the 512px gates through `train.init_from`, with a new step counter, optimizer, and learning-rate schedule.
- The VAE stays in fp32 during training, as in Diffusers' SDXL trainer. Gate heads stay in fp32 inside autocast.

## Metrics and artifacts

- Quality evaluation provides FID, CLIP Score, Precision, and Recall. Precision and recall test membership in any reference point's k-NN ball with k = 3 on Inception features.
- Compositional evaluation uses the official T2I-CompBench++ / UniDet and GenEval adapters.
- Every run records its resolved configuration, environment metadata, seed, and output location.
- Checkpoints contain gate weights, optimizer and learning-rate state, global step, selected layers, and MaskAttn configuration.
- Qualitative generation stores token-wise gate probabilities, hard masks, and post-mask attention maps.
- MaskAttn inference reconstructs its processor configuration from checkpoint metadata, records checkpoint SHA-256, and verifies processor invocation on the final pipeline U-Net.
- Random gates are restricted to the explicit `--allow-untrained-gates` integration smoke; inference, evaluation, and efficiency benchmarking require a trained gate checkpoint.

## Reference environment

| Item | Version / source |
|---|---|
| SDXL checkpoint | `stabilityai/stable-diffusion-xl-base-1.0`, revision `462165984030d82259a11f4367a4eed129e94a7b` |
| Diffusers | `v0.35.1`, commit `0f252be0ed42006c125ef4429156cb13ae6c1d60` |
| PyTorch | `>=2.5`; tested on 2.11.0+cu128 |
| Refiner | not used |

## Test coverage

- Gate dimensions, probability range, hard-gate behavior, and STE gradients.
- Additive mask behavior before softmax and all-masked-row stability.
- Frozen base parameters and trainable masking-head checks.
- Stage and placement selection.
- Gate checkpoint save/load.
- Real Diffusers attention-processor parity with an all-open gate.
- Local SDXL U-Net and pipeline smoke checks.
- End-to-end training, phase continuation, resume, fp16 mixed precision, and mask export on an offline tiny SDXL pipeline.
- Stage selection and gate parameter count on the real SDXL U-Net topology.
- Fréchet distance, precision, recall, and the Clean-FID call signature.
- Final-U-Net attachment, checkpoint metadata compatibility, gate device/dtype consistency, processor call counting, and checkpoint-required inference checks.
