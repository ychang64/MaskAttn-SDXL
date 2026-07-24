# Benchmark and baseline adapters

## Quality evaluation

`experiments/eval_quality.py` first writes images and metadata in the repository's common format, then calls real metric implementations. Install optional dependencies:

```bash
pip install -e '.[metrics]'
```

It requires a local directory of real images paired with the chosen caption split. Do not mix COCO's 3,000-caption protocol with Flickr30k's 500-caption protocol in one aggregate result.

## Official compositional evaluators

Prepare the official T2I-CompBench++/UniDet or GenEval evaluator and pass an explicit command template:

```bash
python experiments/eval_compositional.py \
  --config configs/eval_compositional.yaml \
  --evaluator-command 'python /path/to/official_evaluator.py --images {images} --prompts {prompts} --output {output}'
```

`{images}`, `{prompts}`, and `{output}` are replaced with absolute paths.

## Baselines

Every baseline should emit:

```text
outputs/<method>/
  images/00000.png
  metadata.jsonl
```

Use `python scripts/prepare_baselines.py --name <name>` to print an explicit checkpoint command. Add `--execute` only after checking disk/licensing requirements. `generate_baseline.py` works with local Diffusers-compatible SD-1.5, SD-2.1-base, SDXL, and PixArt pipelines that accept the standard prompt API. Composable Diffusion, Structured Diffusion, and Attend-and-Excite must use their official code adapters and write the same output schema.

`configs/baselines.yaml` records every paper baseline and an explicit checkpoint preparation command where applicable. It never downloads a model automatically. The adapter contract is deliberately narrow: ordered PNGs and a JSONL row containing `index`, `image`, `prompt`, `seed`, and `method` for each sample.

## Efficiency comparisons

`benchmark_efficiency.py` measures warmup-separated repeated generation under one prompt, inference-step count, guidance scale, dtype, and device. ControlNet and GLIGEN comparison settings are recorded in `configs/efficiency_adapters.yaml`; they use their official implementation plus, respectively, a conditioning image or grounding boxes/phrases.
