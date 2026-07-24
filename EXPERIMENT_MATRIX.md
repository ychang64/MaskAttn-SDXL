# Experiment matrix

Every row below has an executable entry point and a configuration. Metrics are written only by the named real evaluator; the paper's values are never embedded as output.

| Paper location | Purpose | Python entry | YAML config | Input data | Output directory / metric file |
|---|---|---|---|---|---|
| Sec. III, Eq. (1)–(5), Fig. 2 | Frozen-SDXL MaskAttn training | `experiments/train_maskattn_sdxl.py` | `configs/train_512.yaml`, `configs/train_1024.yaml` | filtered COCO train2014 cache | `outputs/train_*/`: `config.resolved.yaml`, `environment.json`, loss JSONL, `checkpoints/gate_*.pt` |
| Sec. IV-A training protocol | Build/cache captions containing >=2 noun phrases | `scripts/prepare_coco_captions.py` | CLI | COCO `captions_train2014.json`, images | `data/coco_train2014_multi_noun.jsonl` |
| Table I | COCO val2014 caption quality (3,000) | `experiments/eval_quality.py` | `configs/eval_quality.yaml` | COCO 3k prompt JSONL + matching real-image directory | `outputs/eval_quality_coco/images/`, `metadata.jsonl`, `metrics.json`, `metrics.csv` (FID/CLIP/precision/recall) |
| Table I | Flickr30k caption quality (500) | `experiments/eval_quality.py` | `configs/eval_quality.yaml` + documented CLI overrides | Flickr30k 500 prompt JSONL + real-image directory | `outputs/eval_quality_flickr30k/` with the same files |
| Fig. 3 | Precision/recall evaluation | `experiments/eval_quality.py` | `configs/eval_quality.yaml` | same generated/reference images as Table I | aggregate `metrics.json` / `metrics.csv` |
| Table II | T2I-CompBench++ Spatial / UniDet | `experiments/eval_compositional.py` | `configs/eval_compositional.yaml` | official prompt split + generated images + official evaluator | `outputs/eval_compositional/metrics/` written by the official adapter |
| Table II | GenEval | `experiments/eval_compositional.py` | `configs/eval_compositional.yaml` + `benchmark=geneval` | official GenEval prompts/images/evaluator | configured official evaluator output directory |
| Table III | SDXL vs MaskAttn-SDXL parameter count, CUDA peak memory, latency | `experiments/benchmark_efficiency.py` | `configs/efficiency.yaml` | fixed prompt, local model/checkpoint | `outputs/efficiency/efficiency.json` |
| Table III (comparison targets) | ControlNet / GLIGEN comparison configuration | `experiments/benchmark_efficiency.py` via adapter config | `configs/efficiency_adapters.yaml` | target's required conditioning inputs and local official implementation | target-owned output path |
| Table IV-A | high/mid/low/all stage ablation | `experiments/ablate_gating_stage.py` | `configs/train_512.yaml` + `--stages high mid low all` | same filtered COCO cache | one configured training/checkpoint directory per stage, then Table I/II evaluators |
| Table IV-B | encoder/decoder/full placement ablation | `experiments/ablate_module_placement.py` | `configs/train_512.yaml` + `--placements encoder decoder encoder_decoder` | same filtered COCO cache | one configured training/checkpoint directory per placement, then Table I/II evaluators |
| Fig. 4 | Fixed prompts, same seeds, baseline-vs-MaskAttn grid and token masks | `experiments/generate_qualitative.py` | `configs/generate_qualitative.yaml`, `configs/prompts_qualitative.jsonl` | no external data; local model/checkpoint | `outputs/qualitative/`: images, `comparison_grids/`, metadata, masks, post-mask attention maps |

## Baseline coverage

All baselines use `experiments/generate_baseline.py` where a compatible local pipeline exists, or an external official adapter that writes the exact same `images/` + `metadata.jsonl` contract. `configs/baselines.yaml` provides the explicit preparation/adapter record.

| Paper baseline | Configuration key | Generation path |
|---|---|---|
| SD-1.5 | `sd15` | local Diffusers pipeline |
| SD-2.1-base | `sd21_base` | local Diffusers pipeline |
| SDXL | `sdxl` | local Diffusers pipeline |
| PixArt-α | `pixart_alpha` | local compatible PixArt pipeline |
| PixArt-Σ | `pixart_sigma` | local compatible PixArt pipeline |
| Composable Diffusion | `composable_diffusion` | official external adapter |
| Structured Diffusion | `structured_diffusion` | official external adapter |
| Attend-and-Excite | `attend_and_excite` | official external adapter |

## Stage mapping

This repository maps `down_blocks.1`/`up_blocks.1` to high, `down_blocks.2`/`up_blocks.0` to mid, and `mid_block` to low. The selected paths are persisted in the gate checkpoint sidecar and resolved configuration.
