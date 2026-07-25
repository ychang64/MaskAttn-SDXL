# MaskAttn runtime audit

## Runtime contract

A MaskAttn-SDXL run is accepted only when the final pipeline U-Net owns the selected `MaskAttnProcessor` instances, the checkpoint metadata reconstructs the installed configuration, fp32 gate parameters share the U-Net device, a trained gate checkpoint is loaded, and at least one processor forward call is observed. `load_maskattn_pipeline()` is the single inference entry point enforcing this contract. Checkpoint metadata is the default inference source of truth; `maskattn_override: true` enables strict comparison with a declared config.

`import maskattn_sdxl`, processor attachment, and trained gate checkpoint loading are separate checks.

| Entry point | Actual model | Final U-Net | MaskAttn attachment | Checkpoint source | Random gate policy | Processor execution evidence | Result |
|---|---|---|---|---|---|---|---|
| `scripts/smoke_sdxl.py` | local SDXL Base | `pipe.unet` | none | none | n/a | processor count is zero | baseline SDXL only |
| `scripts/smoke_maskattn_sdxl.py` | local SDXL Base | the same `pipe.unet` used for direct forward and sampling | `load_maskattn_pipeline()` | `--checkpoint` metadata | only `--allow-untrained-gates`; output method is `UNTRAINED_INTEGRATION_ONLY` | `runtime_audit.maskattn_forward_calls > 0` | integration or trained inference |
| `experiments/generate_qualitative.py` | configured SDXL Base | generation pipeline `pipe.unet` | `load_maskattn_pipeline()` through `generate_records()` | `checkpoint` config/CLI | rejected | metadata contains runtime audit and call count | trained checkpoint required |
| `experiments/eval_quality.py` | configured SDXL Base | generation pipeline `pipe.unet` | `load_maskattn_pipeline()` through `generate_records()` | `checkpoint` config/CLI | rejected | generated metadata contains runtime audit | trained checkpoint required |
| `experiments/benchmark_efficiency.py` | configured SDXL Base | benchmark MaskAttn pipeline `pipe.unet` | `load_maskattn_pipeline()` through `_load_pipeline_for_generation()` | `checkpoint` config/CLI | rejected | loader asserts installation; generated benchmark calls processors | trained checkpoint required |
| `experiments/eval_compositional.py` | externally generated images | no pipeline | none | external evaluator input | n/a | evaluator adapter only | no generation occurs here |
| `experiments/generate_baseline.py` | selected baseline | baseline pipeline | none | n/a | n/a | no `MaskAttnProcessor` is installed | baseline only |
| `experiments/train_maskattn_sdxl.py` and ablations | configured SDXL Base | training pipeline `pipe.unet` | `install_maskattn()` before Accelerate preparation | optional training resume checkpoint | permitted for training | device and trainable-parameter assertions | training path |

## Runtime record

MaskAttn image metadata and smoke results record method, checkpoint path and SHA-256, selected layers, gate parameter count, gate device/dtype, processor forward-call count, and checkpoint-loaded state. A forward-call count of zero raises an error in MaskAttn generation and smoke checks.

## Package boundary

The package uses a standard `src/` layout. `pip install -e .` exposes `maskattn_sdxl` for repository-root and external-directory use. A built wheel is also importable from a clean temporary environment. Smoke scripts do not modify `sys.path`.

## Checkpoint availability

Pretrained MaskAttn gate weights are not included in this repository. Training produces `gate_final.pt` with `checkpoint_kind: trained`; qualitative generation, quality evaluation, and efficiency benchmarking require that checkpoint. Checkpoints saved without a training step are marked `test_only` and are rejected by trained inference. The only random-gate path is the explicit `--allow-untrained-gates` integration smoke, whose result is labelled `UNTRAINED_INTEGRATION_ONLY`.
