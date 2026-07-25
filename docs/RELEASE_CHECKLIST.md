# Release checklist

This repository intentionally excludes model weights, datasets, generated images, checkpoints, metric caches, local virtual environments, downloaded PDFs, and the local `third_party/diffusers` audit checkout. Confirm this before each release with `git status --ignored` and an artifact-size review.

Before tagging a public release:

1. Run `python -m pytest -q` and `python -m ruff check src experiments scripts tests` in a clean environment.
2. Run every experimental entry point with `--help` and `--dry-run`.
3. Run `python scripts/smoke_maskattn_sdxl.py --model <local-sdxl-dir> --checkpoint <gate_final.pt> --backward` on a supported host; retain only its text result, not the generated image.
4. Verify `CITATION.cff`, `LICENSE`, model/data/evaluator license notices, and published repository URL/version.
5. Publish experiment configurations, evaluator settings, and generated artifacts together with each release report.

Before tagging a release, confirm the repository URL, version tag, contributor settings, security policy, and issue templates.
