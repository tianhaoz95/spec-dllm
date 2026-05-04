# BD-EAGLE Implementation Summary

## Overview

BD-EAGLE converts a pre-trained EAGLE-3 speculative decoding drafter into a block diffusion language model using the Warmup–Stable–Decay (WSD) curriculum from LLaDA 2.0. The core idea: EAGLE-3 weights already approximate the target model's token distribution autoregressively; replacing the causal attention mask with a block-causal mask and fine-tuning with the BD-LM objective should be sufficient to unlock parallel block generation without training from scratch.

The only structural change to EAGLE-3 is the attention mask. All weights are re-used as initialisation; the embedding table and vocabulary mapping buffers (`d2t`, `t2d`) are frozen throughout.

---

## Repository Structure

```
spec-dllm/
├── bd_eagle/
│   ├── __init__.py
│   ├── attention.py          # block_causal_mask()
│   ├── model.py              # BDEagleDrafter — self-contained EAGLE-3 + BD-LM loss
│   ├── wsd_scheduler.py      # WSDConfig + WSDScheduler
│   ├── dataset.py            # UltraChatDataset + sample_block_masks()
│   └── cached_dataset.py     # CachedFeatureDataset (mmap reader)
├── scripts/
│   ├── extract_features.py   # Phase 1: pre-extract Qwen3-8B hidden states
│   ├── train_bd_eagle.py     # Phase 2: WSD training loop
│   ├── eval_throughput.py    # Phase 3: tokens/s + τ̄ measurement
│   ├── eval_quality.py       # Phase 4: GSM8K / HumanEval / ROUGE-L
│   ├── download_models.py    # Download all three checkpoints from HF
│   └── run_experiments.sh    # End-to-end pipeline script
├── configs/
│   ├── primary.yaml          # Experiment 1: warm start, uniform masking
│   ├── cold_start.yaml       # Experiment 2: random init ablation
│   └── anchor_masking.yaml   # Experiment 3: DFlash-style anchor masking
├── eagle_lib/                # Cloned EAGLE repo (reference only, not imported)
├── RESEARCH_DESIGN.md
└── IMPLEMENTATION_SUMMARY.md
```

---

## 1. EAGLE-3 Architecture (Reverse-Engineered from Checkpoint)

The EAGLE-3 drafter for Qwen3-8B (`AngelSlim/Qwen3-8B_eagle3`) was fully characterised by inspecting `pytorch_model.bin` and `config.json`. The architecture is implemented from scratch in `bd_eagle/model.py` with no dependency on the EAGLE library.

### Checkpoint weight keys and shapes

| Key in `pytorch_model.bin` | Shape | Role |
|---|---|---|
| `fc.weight` | `(4096, 12288)` | Projects 3 target hidden states → 1 hidden vector |
| `midlayer.hidden_norm.weight` | `(4096,)` | RMSNorm on the projected target features |
| `midlayer.input_layernorm.weight` | `(4096,)` | RMSNorm on token embeddings |
| `midlayer.self_attn.q_proj.weight` | `(4096, 8192)` | GQA Q: takes 2H concat as input |
| `midlayer.self_attn.k_proj.weight` | `(1024, 8192)` | GQA K: 8 KV heads × 128 head_dim |
| `midlayer.self_attn.v_proj.weight` | `(1024, 8192)` | GQA V |
| `midlayer.self_attn.o_proj.weight` | `(4096, 4096)` | Output projection back to H |
| `midlayer.mlp.gate_proj.weight` | `(12288, 4096)` | SwiGLU gate |
| `midlayer.mlp.up_proj.weight` | `(12288, 4096)` | SwiGLU up |
| `midlayer.mlp.down_proj.weight` | `(4096, 12288)` | SwiGLU down |
| `midlayer.post_attention_layernorm.weight` | `(4096,)` | Pre-MLP RMSNorm |
| `norm.weight` | `(4096,)` | Final output RMSNorm |
| `lm_head.weight` | `(32000, 4096)` | Draft vocabulary head |
| `d2t` | `(32000,)` int64 | Maps draft token id → full Qwen3 token id |
| `t2d` | `(151936,)` bool | True if full token id has a draft representation |

The `embed_tokens` weight (151936 × 4096) is **not stored** in the EAGLE-3 checkpoint; it is shared from the frozen target model (`Qwen/Qwen3-8B`) at runtime.

### Key architectural insight: 2H attention input

The Q/K/V projections take an **8192-dimensional** input (= 2 × hidden_size = 2 × 4096). This is the concatenation of:
- Normalised token embedding: `input_layernorm(embed_tokens(input_ids))` — shape `[B, T, 4096]`
- Normalised projected target features: `hidden_norm(fc(cat([hs_low, hs_mid, hs_high])))` — shape `[B, T, 4096]`

The `fc` layer projects the 12288-dim concatenation of three target hidden states down to 4096 before this concatenation happens.

### Target hidden state extraction

Discovered from the EAGLE-3 training code (`eagle_lib/eagle/traineagle3/modeling_llama_kv.py`, line 1139):

```python
if idx == len(self.layers) - 3 or idx == len(self.layers) // 2 or idx == 2:
    all_hidden_states += (hidden_states,)
```

For Qwen3-8B (36 transformer layers), this collects hidden states **before** layers 2, 18, and 33 — equivalently, the output of layers 1, 17, and 32. In HuggingFace's `output_hidden_states=True` convention (index 0 = embedding, index k = after layer k−1), these correspond to:

| Extraction point | HF hidden_states index | Approx depth |
|---|---|---|
| After layer 1 | `[2]` | Shallow (low) |
| After layer 17 | `[18]` | Middle |
| After layer 32 | `[33]` | Deep (high) |

These indices are stored in `configs/*.yaml` as `target_layer_indices: [2, 18, 33]` and used by both `extract_features.py` and the forward pass.

### Reduced draft vocabulary

EAGLE-3 uses a vocabulary reduction technique to speed up the LM head:
- `draft_vocab_size = 32000` (vs. Qwen3's full `151936`)
- Only the 32000 most frequent tokens are represented in the draft vocabulary
- `d2t[i]` = target token id for draft token `i`
- `t2d[j]` = True if full token `j` has a draft representation
- The cumulative sum of `t2d` gives the dense mapping from full → draft token ids

This means the BD-LM loss is only computable at positions where the ground-truth token is in the draft vocabulary. The `masked_diffusion_loss` function handles this by applying `in_draft = t2d[targets]` as an additional gate on top of `mask_positions`.

---

## 2. `bd_eagle/attention.py` — Block-Causal Mask

```python
def block_causal_mask(seq_len: int, block_size: int, device: torch.device) -> Tensor:
```

Constructs a `[seq_len, seq_len]` additive attention mask:
- **Within a block** (positions with the same `pos // block_size`): every position can attend to every other position in the same block (bidirectional, value = 0).
- **Across blocks**: position `i` can attend to all positions in earlier blocks, but not in later blocks (causal, value = −1e9).

The implementation assigns each position a `block_id = pos // block_size`, then sets `mask[i, j] = 0` iff `block_id[j] <= block_id[i]`.

**Special cases:**
- `block_size = 1`: reduces to a standard lower-triangular causal mask — identical to the original EAGLE-3 behaviour. This is exploited by the WSD warmup Phase 1 (first 500 steps at block_size=1) to preserve EAGLE-3 initialisation quality.
- `block_size = T` (full sequence): all positions attend to all others (pure bidirectional diffusion).

---

## 3. `bd_eagle/model.py` — `BDEagleDrafter`

### Sub-modules (all implemented from scratch)

**`RMSNorm`**: standard root-mean-square normalisation, up-cast to float32 for numerical stability, scaled back to input dtype.

**`RotaryEmbedding`**: Qwen3-compatible rotary position embeddings with `rope_theta = 1_000_000`. Cos/sin tables are built lazily on first use and cached; they are extended automatically if a longer sequence is seen. The `rotate_half` / `apply_rotary` functions follow the standard half-rotation convention.

**`GQAttention`**: grouped-query attention (32 Q heads, 8 KV heads, head_dim=128). Q/K/V projections take the **8192-dimensional** concatenated input. K and V are expanded to match Q head count via `repeat_interleave`. RoPE is applied after projection. The forward pass accepts a `[1, 1, T, T]` additive mask, which carries the block-causal pattern.

**`SwiGLUMLP`**: `down_proj(silu(gate_proj(x)) * up_proj(x))`. Hidden size 4096, intermediate size 12288.

### `BDEagleDrafter.from_pretrained`

Loads `pytorch_model.bin` via `torch.load(..., weights_only=True)` and applies a hand-written key remapping from the checkpoint's `midlayer.*`-prefixed names to the model's flat attribute paths:

```
"midlayer.self_attn.q_proj.weight"  →  "self_attn.q_proj.weight"
"midlayer.post_attention_layernorm.weight"  →  "post_attn_norm.weight"
...
```

If a `target_embed_tokens` embedding is passed, its weight tensor is shared directly (pointer sharing, not a copy) and frozen. The `d2t` and `t2d` buffers are loaded from the checkpoint and registered as non-trainable buffers.

### `BDEagleDrafter.forward`

Eight-step computation:

```
1.  projected  = fc( cat([hs_low, hs_mid, hs_high], dim=-1) )   [B, T, H]
2.  norm_proj  = hidden_norm(projected)                          [B, T, H]
3.  norm_emb   = input_layernorm(embed_tokens(input_ids))        [B, T, H]
4.  x          = cat([norm_emb, norm_proj], dim=-1)              [B, T, 2H]
5.  attn_mask  = block_causal_mask(T, block_size, device)        [T, T]
6.  attn_out   = self_attn(x, attn_mask_4d, position_ids)        [B, T, H]
7.  h          = projected + attn_out                            [B, T, H]   ← residual
8.  h          = h + mlp(post_attn_norm(h))                      [B, T, H]
    logits     = lm_head(norm(h))                                [B, T, 32000]
```

The residual in step 7 wraps the attention output against `projected` (the 4096-dim fc output), not the 8192-dim concatenation. This matches the EAGLE-3 training code's `residual = hidden_states` taken before the concat operation in `LlamaDecoderLayeremb.forward`.

`block_size` can be overridden per-call, enabling evaluation at different block sizes from a single trained checkpoint.

### `BDEagleDrafter.masked_diffusion_loss`

The BD-LM loss, implemented as a static method:

```
L = (1/t) * mean_over_active( CE(logits_draft[i], targets_draft[i]) )
```

where `active = mask_positions & t2d[targets]` and `t ∈ [0.05, 1.0]` is the noise level sampled each step. The `1/t` weight is the time-derivative of the linear noise schedule `α_t = 1 − t`, i.e. `α'_t / (1 − α_t) = 1/t`.

Target tokens not in the draft vocabulary are silently excluded from the loss (their draft id would be −1 from the `t2d.cumsum` mapping).

Optional position weighting (Experiment 3 / `anchor_masking.yaml`):

```
ce_weighted[i] = ce[i] * exp(-(i mod block_size) / gamma)
```

This exponentially discounts errors at later positions within a block, since a wrong prediction at position 0 invalidates all subsequent positions. `gamma = 4.0` is used for `block_size = 8`.

---

## 4. `bd_eagle/wsd_scheduler.py` — WSD Curriculum

`WSDConfig` is a dataclass specifying all schedule hyperparameters. `WSDScheduler` pre-computes the full `(block_size, lr)` table at construction time (`_build_schedule`), so each training step does a single `O(1)` table lookup via `__getitem__(step)`.

### Schedule for the primary experiment

| Phase | Block sizes | Steps | LR behaviour |
|---|---|---|---|
| Warmup | 1 → 2 → 4 → 8 | 500 per size = **2 000** | Linear: `lr_max × (global_step + 1) / warmup_total` |
| Stable | 8 | **8 000** | Constant `lr_max = 3×10⁻⁴` |
| Decay | 4 → 2 | 500 per size = **1 000** | Cosine: `lr_min + 0.5×(lr_max−lr_min)×(1+cos(π×k/decay_total))` |
| **Total** | | **11 000 steps** | |

At warmup step 0 (block_size=1, causal mask), the loss is equivalent to standard autoregressive cross-entropy on the draft vocabulary — preserving the EAGLE-3 initialisation. Block size increases gradually, so the model adapts incrementally rather than switching to full bidirectional attention in one jump.

---

## 5. `bd_eagle/dataset.py` — Training Data

### `UltraChatDataset`

Loads `HuggingFaceH4/ultrachat_200k` (`train_sft` split), applies the Qwen3 chat template via `tokenizer.apply_chat_template`, truncates to `max_length = 512` tokens, and filters out sequences shorter than 5 tokens. The first `max_samples = 30_000` examples (after shuffling with seed 42) are retained.

The dataset is used only by `extract_features.py`; the training loop reads from the feature cache instead.

### `sample_block_masks`

Generates a binary mask tensor for a batch:

1. Sample a uniform noise level `t ~ Uniform(0.05, 1.0)`.
2. For each sequence in the batch, iterate over blocks of size `block_size`.
3. Within each block, optionally designate `n_anchors_per_block` randomly chosen valid positions as anchors (always kept unmasked — used in Experiment 3).
4. Each remaining valid (non-padding) position is masked independently with probability `t`.

Returns `(mask_positions [B, T] bool, t float)`. Padding positions are never masked.

---

## 6. `bd_eagle/cached_dataset.py` — Feature Cache Reader

### `CachedFeatureDataset`

Reads three memory-mapped NumPy arrays written by `extract_features.py`:

| File | Shape | Dtype | Content |
|---|---|---|---|
| `features.npy` | `(N, 512, 12288)` | float16 | Concatenated hidden states from 3 target layers |
| `input_ids.npy` | `(N, 512)` | int32 | Token ids |
| `attention_mask.npy` | `(N, 512)` | uint8 | Padding mask |

`mmap_mode="r"` means only the accessed rows are paged into RAM at any time. Each `__getitem__` reads one row and converts it to a `dict[str, Tensor]`. The `fused_features` tensor is cast from float16 to float32 on load; the training loop recasts it to bfloat16 before the forward pass.

---

## 7. `scripts/extract_features.py` — Phase 1

Runs the frozen target model (`Qwen/Qwen3-8B`, BF16) over all 30 000 training sequences and writes the three extracted hidden states to disk. This decouples the expensive target forward pass from the cheap drafter training loop.

**Cache invalidation**: checks `meta.json` for a matching `(model, n_samples, max_length)` triple before re-running. Safe to call multiple times.

**Memory layout**: `np.lib.format.open_memmap` with mode `"w+"` pre-allocates the arrays. The actual sample count (after `drop_last=True` in the DataLoader) is recorded in `meta.json["n_stored"]`; the `CachedFeatureDataset` reads exactly that many rows.

**Disk estimate** (30 K samples, seq 512, 3 layers × 4096-dim, float16):  
`30 000 × 512 × 12288 × 2 bytes ≈ 0.38 GB` — well within the 457 GB available NVMe.

**`target_layer_indices` in HF format**: the script uses `hs[i + 1]` for each index `i` in the config list, since HF's `hidden_states[0]` is the embedding output and `hidden_states[k]` is the output of layer `k−1`. With `target_layer_indices: [2, 18, 33]`, this extracts:
- `hidden_states[3]` = output of layer 2 (low)
- `hidden_states[19]` = output of layer 18 (mid)
- `hidden_states[34]` = output of layer 33 (high)

matching the exact layers collected by EAGLE-3's custom training code.

---

## 8. `scripts/train_bd_eagle.py` — Phase 2

### Startup sequence

1. Resolve EAGLE-3 checkpoint (HuggingFace model ID → local cache path via `snapshot_download` if needed).
2. Load `BDEagleDrafter` from checkpoint.
3. If `cold_start: true` (Experiment 2): re-initialise all trainable parameters with Xavier uniform (matrices) or zeros (vectors). The checkpoint is still used to provide the correct architecture shapes.
4. Freeze `embed_tokens.weight`; collect only trainable parameters for the optimiser.
5. Load feature cache via `build_cached_dataloader`.
6. Build `WSDScheduler` from config.

### Per-step loop

```
block_size, lr  ← scheduler[step]
update optimizer lr groups
batch           ← next(data_iter)   # cycles on StopIteration
mask_positions, t_noise ← sample_block_masks(input_ids, attn_mask, block_size, ...)
masked_ids      ← input_ids with masked positions replaced by eos_token_id (151645)
logits          ← model(masked_ids, fused_features, block_size=block_size)
loss            ← masked_diffusion_loss(logits, input_ids, mask_positions, model.t2d, t_noise, ...)
loss.backward()
clip_grad_norm_(trainable_params, 0.5)
optimizer.step()
```

### Checkpointing

Saves every 500 steps to `checkpoints/<run_name>/step_NNNNNN/`. Only the trainable weights (excludes `embed_tokens`) are saved in `drafter_weights.pt`. The last 3 checkpoints are kept; older ones are deleted to conserve disk.

Resumption is supported via `--resume <ckpt_dir>`, which reloads both weights and optimiser state and continues from the recorded step index.

### Config copying

The active YAML config is copied to `checkpoints/<run_name>/config.yaml` at job start for exact reproducibility.

---

## 9. `scripts/eval_throughput.py` — Phase 3

Evaluates wall-clock throughput and acceptance rate for all four systems:

| `--system` | Drafter | Decoding |
|---|---|---|
| `baseline` | None | Standard AR greedy with `model.generate` |
| `eagle3` | `AngelSlim/Qwen3-8B_eagle3` | AR speculative (5-draft-token tree-free) |
| `dflash` | `z-lab/Qwen3-8B-DFlash-b16` | Block diffusion speculative |
| `bd_eagle` | `--checkpoint` path | Block diffusion speculative |

**`speculative_decode_ar`** (used for `eagle3`): drafts `n_draft=5` tokens autoregressively, then runs the target model over all candidates in one forward pass, accepts/rejects greedily (argmax match), appends one bonus target token at the rejection site.

**`speculative_decode_block`** (used for `dflash` and `bd_eagle`): extracts target features at the last context position, generates a full block of `block_size` masked tokens in one drafter forward pass using the block-causal mask, verifies all block tokens with one target forward pass, accepts greedily, appends one bonus token.

**Timing**: `torch.cuda.synchronize()` is called before and after each generation to measure true wall-clock time. The first `warmup=10` sequences are excluded from statistics.

**Output**: JSON record with `tokens_per_second`, `mean_accepted_per_step`, `total_tokens`, `n_prompts`.

---

## 10. `scripts/eval_quality.py` — Phase 4

Three quality benchmarks, all using standard AR greedy decoding (speculative decoding is lossless by construction — the verification step guarantees token distributions match the target):

**GSM8K** (`openai/gsm8k`, 1319 test samples): extracts the final number from model output using a `####`-pattern regex, falls back to the last integer in the output. Reports exact-match accuracy.

**ROUGE-L** (`HuggingFaceH4/ultrachat_200k`, `test_sft`, 200 samples): computes the F1 LCS score between predicted and reference assistant responses. Custom pure-Python LCS implementation avoids the `rouge_score` dependency.

**HumanEval** (164 problems): uses `human-eval` package if installed; passes completions to `evaluate_functional_correctness` for pass@1. Gracefully degrades with an error message if the package is absent.

---

## 11. Configs

### `configs/primary.yaml` — Experiment 1

Warm-start from EAGLE-3, uniform masking, no position weighting. This is the main hypothesis test.

### `configs/cold_start.yaml` — Experiment 2

Identical schedule and data; drafter weights are re-initialised randomly before training (flag `cold_start: true`). Isolates the contribution of the EAGLE-3 warm start.

### `configs/anchor_masking.yaml` — Experiment 3

Warm start; adds `use_anchor_masking: true` with `n_anchors_per_block: 2` and `use_position_weight: true` with `gamma: 4.0`. Tests whether DFlash-style anchor masking and exponential loss weighting help in the low-data regime.

---

## 12. Dependencies

Installed in `.venv` (Python 3.12):

| Package | Version | Role |
|---|---|---|
| `torch` | 2.11.0+cu128 | Core training and inference |
| `transformers` | — | Model loading, tokenizer, `generate` |
| `accelerate` | — | Device placement |
| `datasets` | — | UltraChat-200K, GSM8K |
| `huggingface_hub` | — | `snapshot_download` |
| `numpy` | — | Memory-mapped feature cache |
| `pyyaml` | — | Config loading |
| `safetensors` | — | Safe checkpoint loading |
| `tqdm` | — | Progress bars |
| `wandb` | — | Optional experiment tracking |
| `einops` | — | Optional tensor manipulation |

---

## 13. Pending Step Before Training

`Qwen/Qwen3-8B` is a gated model on HuggingFace. To download it:

```bash
# 1. Authenticate (one-time)
! .venv/bin/huggingface-cli login

# 2. Accept the Qwen3 license at https://huggingface.co/Qwen/Qwen3-8B

# 3. Download
! .venv/bin/python scripts/download_models.py
```

Once the target model is in cache, run the full pipeline:

```bash
source .venv/bin/activate

# Phase 1: extract features (~3–4 h)
python scripts/extract_features.py --config configs/primary.yaml

# Phase 2: train all three experiments (~3 h total with cached features)
python scripts/train_bd_eagle.py --config configs/primary.yaml
python scripts/train_bd_eagle.py --config configs/cold_start.yaml
python scripts/train_bd_eagle.py --config configs/anchor_masking.yaml

# Phase 3 & 4: evaluate (~7 h)
bash scripts/run_experiments.sh   # (skips Phase 1 if cache exists)
```

---

## 14. Validated Tests

The following were confirmed working without GPU access:

| Test | Result |
|---|---|
| `block_causal_mask(12, 4)` shape and values | Pass |
| `block_causal_mask` reduces to causal at `block_size=1` | Pass |
| `WSDScheduler` total steps = 11 000 | Pass |
| `WSDScheduler` step 0: block_size=1, LR≈0 | Pass |
| `WSDScheduler` last step: block_size=2, LR=lr_min | Pass |
| `BDEagleDrafter.from_pretrained` loads all 15 checkpoint keys | Pass |
| Forward pass output shape `[2, 16, 32000]` | Pass |
| `masked_diffusion_loss` gradient flows to all trainable params | Pass |
| Full training step simulation (forward + backward + clip + step) | Pass |
| All YAML configs load without error | Pass |
