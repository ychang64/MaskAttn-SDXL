# Paper audit

This record compares the code with the IJCNN 2026 camera-ready paper. It has three parts: what the paper fixes, what the paper leaves open, and what has not been verified.

## Settings the paper fixes

| Paper | Code | Status |
|---|---|---|
| Eq. (1): `G-hat = sigmoid(f(X, e_t))` on the current feature map and text tokens | `MaskGateHead`, called from `MaskAttnProcessor` | matches |
| Head shared across timesteps; no timestep input | `MaskGateHead.forward` takes features and tokens only | matches |
| Head attached "at each cross-attention site" | `gate_sharing: block`, one head per spatial transformer | matches; see the open question below |
| Eq. (2): threshold 0.5 with a straight-through estimator | `hard_gate_with_ste`, `masked_attention` | matches |
| Eq. (3): 0 or a large negative constant | `additive_mask`, `negative_value: -10000.0` | matches |
| Eq. (4): `Softmax(QK^T / sqrt(d) + M) V`, mask broadcast across heads | `masked_attention` | matches; test compares it with PyTorch's float-mask attention |
| Eq. (5): residual two-layer GELU FFN | SDXL's frozen `BasicTransformerBlock` feed-forward network | matches in form; SDXL applies a LayerNorm and adds it to `X + A`. No new FFN is added, because the paper trains only the masking heads |
| SDXL backbone, VAE, and text encoders frozen | `freeze_base_parameters`, `assert_only_maskattn_trainable` | matches |
| Default placement: mid-resolution blocks | `stage: mid`, `placement: encoder_decoder` | matches |
| 3.6M added parameters (Sec. IV-C) | 3,600,645 with the default stage on the real SDXL topology | matches; a test pins this count |
| COCO train2014, 200k captions with at least two noun phrases | `scripts/prepare_coco_captions.py --limit 200000` | matches; captions are sampled in seeded random order |
| 512px: 100k steps, effective batch 16, AdamW `1e-4`, weight decay 0.01, betas 0.9/0.999, 1k warmup, cosine, clip 1.0, mixed precision | `configs/train_512.yaml` | matches |
| 1024px: 10k more steps at batch size 8 | `configs/train_1024.yaml` | matches; effective batch 8 from 4 x 2 accumulation |
| COCO val2014 3,000 captions, Flickr30k 500 captions | `scripts/prepare_eval_prompts.py` | matches |
| Stage ablation high/mid/low/all; placement encoder/decoder/full | `ablate_gating_stage.py`, `ablate_module_placement.py` | matches |

## Settings the paper leaves open

These are assumptions of this code. They are not claims about the code that produced the paper's numbers.

| Open question | Choice here | Basis |
|---|---|---|
| Gate head architecture | Separate MLP branches for locations and tokens, scored by a scaled dot product | The paper allows "a shallow conv/MLP head" |
| Head width and sharing | `gate_hidden_dim: 192`, `gate_sharing: block` | Fitted, not derived: five mid-stage spatial transformers at 720,129 parameters each give 3,600,645. "Cross-attention site" could instead mean each of the 50 layers, which has no natural head size that reaches 3.6M |
| Gate initialisation | Open everywhere (`gate_init_logit: 2.0`) | A plug-in should start as plain SDXL. A zero bias would mask about half of all token-location pairs at step 0 |
| Training loss | Standard noise-prediction MSE, no sparsity term | The paper states no loss |
| Where the estimator sits | Forward uses the hard mask; backward uses `Softmax(logits + log G-hat)` | On the logits, a closed gate would have zero gradient and could not reopen |
| Value of the negative constant | `-10000.0` | The paper says only "a large negative constant" |
| A location that loses every token | Keep its most probable token | Needed for `-inf`. With a finite constant the softmax would cancel it and leave the row unmasked |
| Stage names on SDXL | high: `down_blocks.1`/`up_blocks.1`; mid: `down_blocks.2`/`up_blocks.0`; low: `mid_block` | SDXL has cross-attention at two resolutions only, so mid and low run at the same size and differ by depth |
| Sampler steps and guidance | 30 steps, guidance 7.5, the SDXL default scheduler | The paper says only that sampling is matched across methods |
| CLIP Score model | ViT-B-32 `laion2b_s34b_b79k`; configurable through `clip_model` and `clip_pretrained` | Not stated. Scores differ between CLIP weights, so check this before comparing with Table I |
| Precision/recall features and k | Clean-FID Inception features, k = 3 | Not stated. The original metric used VGG-16 features |
| FID reference set | The real images paired with the sampled captions | Not stated. Precomputed statistics of the full split would give different values |

## Not verified

- No run has used real SDXL weights, and no gates have been trained.
- The paper's tables and figures have not been reproduced with this code.
- The T2I-CompBench++ and GenEval adapters have not been run against the official evaluators.
- Composable Diffusion, Structured Diffusion, and Attend-and-Excite have no generation path here.
- `stabilityai/stable-diffusion-2-1-base` returned HTTP 401 to an anonymous request, so that baseline may need a Hugging Face login or another source.
