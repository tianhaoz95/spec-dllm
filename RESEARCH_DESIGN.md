# BD-EAGLE: Block Diffusion Adaptation of EAGLE-3 Drafters via WSD Fine-Tuning

**Date:** 2026-05-03  
**Hardware:** NVIDIA DGX Spark (GB10 Grace Blackwell Superchip, 128 GB unified LPDDR5X, 20-core ARM Grace CPU, 457 GB NVMe)  
**48-hour budget:** ~10 h training · ~4 h evaluation · ~6 h debugging buffer · ~28 h spare

---

## 1. Motivation

Speculative decoding accelerates LLM inference by letting a cheap drafter propose candidate tokens that the target model verifies in one parallel forward pass. Two drafter paradigms now compete:

| Paradigm | Example | Draft mechanism | Latency bottleneck |
|---|---|---|---|
| Autoregressive (AR) | EAGLE-3 | Token-by-token, causal tree | Sequential draft steps |
| Block diffusion | DFlash | All block tokens in one pass | Single draft forward pass |

DFlash trains its block-diffusion drafter **from scratch** using a bespoke masking curriculum (~800 K samples, purpose-built architecture). LLaDA 2.0 separately showed that a **Warmup–Stable–Decay (WSD)** curriculum can convert a pre-trained AR model into a masked diffusion LM via continual fine-tuning — with no architectural changes beyond the attention mask.

**Hypothesis:** Initialising the block-diffusion drafter from a pre-trained EAGLE-3 checkpoint and applying WSD fine-tuning achieves comparable acceptance rates to DFlash using ≤30 K training samples (~40× fewer than DFlash), because EAGLE-3 already approximates the target distribution autoregressively.

---

## 2. Background

### 2.1 EAGLE-3

- **Drafter**: one Transformer decoder layer + a fully-connected fusion layer that compresses hidden states extracted from three depth levels of the target model (low / mid / high) from `3k → k` dims, where `k` is the target hidden size.
- **Training**: cross-entropy over draft token predictions only (no feature-prediction loss). "Training-time test" simulates multi-step rollout: step 2 onward feeds the drafter's own previous output, not ground-truth target features, so the model learns to handle its own compounding errors.
- **Draft**: dynamic tree (up to 60 nodes, depth 6) built with top-k expansion; verified by the target via tree attention.
- **Key checkpoint used here**: `AngelSlim/Qwen3-8B_eagle3`

### 2.2 DFlash

- **Drafter**: 5 Transformer layers; target hidden states (from 5 uniformly sampled layers, linearly projected) are injected into the **Key and Value** of every draft layer — persistent conditioning, not input-only.
- **Block generation**: generates an entire block of `b = 16` tokens in a single forward pass (one denoising step). Anchor masking during training: random "anchor" tokens are left unmasked; everything else in the block is masked. Loss is exponentially weighted by position (`exp(−(k−1)/γ)`) to up-weight early positions.
- **Training cost**: ~800 K samples (Nemotron Post-Training V2 + CodeAlpaca), 6 epochs, lr = 6×10⁻⁴.
- **Key checkpoint for comparison**: `z-lab/Qwen3-8B-DFlash-b16`

### 2.3 LLaDA 2.0 WSD

WSD converts a pre-trained AR model into a Block Diffusion LM (BDLM) in three phases:

| Phase | Block size progression | Purpose |
|---|---|---|
| Warmup | 1 → 4 → 32 → 64 → 4096 | Progressively expand the bidirectional receptive field |
| Stable | 4096 (= full sequence MDLM) | Deeply internalise the diffusion objective |
| Decay | 4096 → 32 | Shrink to efficient inference block size with KV caching |

The only structural change is the **attention mask**: causal within Phase 1, block-causal (bidirectional within each block, causal across blocks) in Phases 2–3. No weight changes to projections or layers. Checkpoint merging across top-k best checkpoints stabilises phase transitions.

### 2.4 Block Diffusion LM (BD-LM) Objective

For a block `k` of length `b`, with positions `i` randomly masked at rate `α_t`:

```
L_BDLM = −E_t [ α'_t / (1 − α_t) · Σ_i 1[x_{t,k}^i = MASK] · log p_θ(x_{0,k}^i | x_{0,<k}, x_{t,k}) ]
```

where `x_{0,<k}` is the causal context of all previous (clean) blocks, attended to with standard KV caching.

---

## 3. Proposed Method: BD-EAGLE

### 3.1 Core Idea

We adapt the WSD curriculum from LLaDA 2.0 **to the EAGLE-3 drafter only** (the target model stays frozen as a standard AR LLM and is used solely for feature extraction and token verification). The drafter block size ramps from 1 to 8 and back, which is appropriate for speculative decoding windows (DFlash uses 16; we start conservative to preserve acceptance rate).

### 3.2 Architecture

Start from `AngelSlim/Qwen3-8B_eagle3` (EAGLE-3 drafter for Qwen3-8B). This drafter contains:
- FC fusion layer: `[h_low; h_mid; h_high] ∈ R^{3k} → R^k`, where `k = 4096` for Qwen3-8B.
- One Transformer decoder layer with causal self-attention (`q_dim = k = 4096`).
- Shared embedding table and LM head with the frozen target model.

**Modification — attention mask only:**  
Replace the standard causal mask with a block-causal mask:

```
Mask_{ij} = {
  0 (attend)    if  block(i) > block(j),          # causal across blocks
  0 (attend)    if  block(i) == block(j),          # bidirectional within block
  -inf (ignore) otherwise
}
```

Concretely: for block size `b`, positions `[nb, (n+1)b)` form block `n`. Position `i` can attend to all positions `j ≤ i + (b − 1 − (i mod b))` within the same block, and all positions in previous blocks.

No new parameters are introduced. The FC fusion layer and transformer layer weights are initialised from the EAGLE-3 checkpoint.

### 3.3 WSD Training Schedule (BD-EAGLE)

We compress LLaDA 2.0's 1→4096→32 trajectory to a range suitable for speculative drafting:

| Phase | Block sizes | Steps | LR schedule |
|---|---|---|---|
| Warmup | 1 → 2 → 4 → 8 | 500 per size = 2 000 total | Linear warmup 0 → 3×10⁻⁴ |
| Stable | 8 | 8 000 | Constant 3×10⁻⁴ |
| Decay | 8 → 4 → 2 | 500 per size = 1 500 total | Cosine decay → 3×10⁻⁵ |
| **Total** | | **11 500 steps** | |

At block_size = 1 (first 500 steps), the mask is the standard causal EAGLE-3 mask and the loss is standard CE — the checkpoint is being refined, not shocked.

**Loss function:**  
Standard masked-diffusion cross-entropy (BD-LM objective above). Masking rate `t ~ Uniform(0, 1)`, masking probability per token = `t`. At block_size = 1, this degenerates to standard AR CE, preserving EAGLE-3 initialisation quality.

**Anchor masking (DFlash-style data augmentation, optional ablation):**  
Randomly select `n_anchors = max(1, b // 4)` positions in each block as "anchors" (unmasked regardless of `t`). Anchors act as conditioning tokens for predicting the masked positions. This mirrors inference behaviour at block_size = 8 and provides implicit data augmentation. We include this as an optional ablation run (Experiment 3).

### 3.4 Feature Extraction

We adopt the EAGLE-3 multi-layer fusion strategy (not DFlash's per-layer KV injection) to minimise architectural changes:
- Extract hidden states from layers {6, 16, 28} of Qwen3-8B (low / mid / high, indices 0-indexed out of 36 layers).
- Concatenate: `[h_6; h_{16}; h_{28}] ∈ R^{3×4096}`.
- Project via the frozen FC fusion layer to `R^{4096}`.
- This vector is prepended as a conditioning token to the block input (matching EAGLE-3 inference protocol).

The target model runs a frozen forward pass; only the drafter weights (FC layer + 1 Transformer layer) are updated.

---

## 4. Model and Dataset Selection

### 4.1 Target Model

**Qwen3-8B-Instruct** (`Qwen/Qwen3-8B-Instruct`)

Rationale:
- EAGLE-3 checkpoint available: `AngelSlim/Qwen3-8B_eagle3`.
- DFlash checkpoint available: `z-lab/Qwen3-8B-DFlash-b16` — enables direct comparison on identical target model.
- Memory footprint in BF16: ~16 GB — well within 128 GB unified memory.
- Training the drafter (single layer, ~400 M params) against this target is fast on the GB10.

### 4.2 EAGLE-3 Drafter Initialisation

`AngelSlim/Qwen3-8B_eagle3` (HuggingFace)

### 4.3 Training Dataset

**UltraChat-200K** (subset: 30 000 samples)

- HuggingFace: `HuggingFaceH4/ultrachat_200k` (train split, first 30 K)
- Same corpus used in EAGLE-3 training (the 464 K UltraChat portion). Re-using it preserves the distribution alignment the EAGLE-3 weights were already trained on — important for the warm-start hypothesis.
- At ~512-token average response length: ~15.4 M tokens total. This is a fast run by any standard.

### 4.4 Evaluation Datasets

| Benchmark | Splits | Measures |
|---|---|---|
| MT-Bench | 80 questions | General chat quality (Judge-LLM score) |
| GSM8K | test (1319) | Arithmetic reasoning (exact match) |
| HumanEval | 164 problems | Code generation (pass@1) |
| UltraChat held-out | 2 000 samples | ROUGE-L, perplexity of accepted tokens |

These mirror the DFlash paper's evaluation suite, enabling apples-to-apples comparison.

---

## 5. Experiments

### Experiment 1 (Primary): BD-EAGLE vs EAGLE-3 vs DFlash

Train BD-EAGLE following the WSD schedule in §3.3. Evaluate all three systems on the four benchmarks with the Qwen3-8B-Instruct target.

**Metrics collected at inference time:**
- Mean accepted tokens per step (τ̄): the key acceptance-rate measure.
- Wall-clock throughput (tokens/second) at batch size 1.
- Quality: task accuracy / ROUGE-L — must not degrade vs. standard greedy decoding.
- Drafter FLOPs per accepted token (efficiency measure).

**Baselines:**
1. **Greedy AR** (Qwen3-8B-Instruct, no speculative decoding) — throughput floor.
2. **EAGLE-3** (`AngelSlim/Qwen3-8B_eagle3`) — AR tree drafter, our warm-start source.
3. **DFlash** (`z-lab/Qwen3-8B-DFlash-b16`) — trained-from-scratch block diffusion drafter.
4. **BD-EAGLE (ours)** — WSD fine-tuned from EAGLE-3.

### Experiment 2 (Ablation): Warm Start vs Cold Start

Re-run the WSD training with the drafter initialised from **random weights** instead of EAGLE-3. Identical training budget and schedule. Measures how much of BD-EAGLE's performance comes from the EAGLE-3 initialisation vs. the WSD training itself.

Expected result: the EAGLE-3 warm start should significantly reduce the number of steps needed to reach a given acceptance rate, validating the core hypothesis.

### Experiment 3 (Ablation): Anchor Masking vs Uniform Masking

Two training runs, identical except:
- Run A: pure uniform masking (`t ~ Uniform(0,1)`, no anchors).
- Run B: anchor masking (`n_anchors = b // 4` positions always unmasked).

Measures the DFlash-style anchor masking benefit in our lower-data regime.

### Experiment 4 (Scaling): Block Size Sensitivity

Evaluate BD-EAGLE at block sizes b ∈ {2, 4, 8} at inference time (the model trained with max block size 8 supports any smaller block size by narrowing the attention mask). Plot τ̄ vs. b vs. tokens/second to identify the Pareto-optimal block size for this hardware.

---

## 6. Hardware Feasibility Analysis

### 6.1 GB10 Specifications (DGX Spark)

| Resource | Value |
|---|---|
| GPU | NVIDIA GB10 Grace Blackwell Superchip |
| Unified memory | 128 GB LPDDR5X (CPU + GPU shared) |
| Peak AI perf | ~1 PFLOP FP4 / ~500 TFLOP FP8 / ~250 TFLOP BF16 |
| Memory bandwidth | ~273 GB/s |
| CPU | 20-core ARM Neoverse (Grace), up to 3.9 GHz |
| NVMe free | 457 GB |
| CUDA | 13.0, Driver 580.142 |

The GPU reports `Memory-Usage: Not Supported` in `nvidia-smi` because the GB10's unified memory architecture does not maintain a discrete VRAM pool — the entire 128 GB is directly addressable by both CPU and GPU.

### 6.2 Memory Budget

| Component | Size (BF16) |
|---|---|
| Qwen3-8B-Instruct (frozen, BF16) | ~16 GB |
| EAGLE-3 drafter (400 M params) | ~0.8 GB |
| Drafter gradients | ~0.8 GB |
| AdamW optimizer states (2× weights) | ~1.6 GB |
| Activations + KV cache (seq 512, batch 8) | ~4 GB |
| DataLoader buffers | ~0.5 GB |
| **Total** | **~24 GB** |

128 GB available — **comfortable headroom** (~5× margin), no quantisation required.

### 6.3 Training Time Estimate

Per training step (batch size 8, sequence length 512):

| Component | FLOPs | Time @ 250 TFLOP/s BF16 (40% efficiency) |
|---|---|---|
| Qwen3-8B frozen forward (feature extraction) | 6 × 8×10⁹ × 512 × 8 = 1.97×10¹⁴ | ~2.0 s |
| Drafter forward + backward (400 M params) | 3 × 6 × 4×10⁸ × 512 × 8 = 2.95×10¹³ | ~0.3 s |
| **Total per step** | | **~2.3 s** |

11 500 steps × 2.3 s = **~7.4 hours**.

> Note: the Qwen3-8B forward dominates. Since its weights are frozen, we can pre-compute and cache the feature vectors to disk at the start of training. With 30 K sequences × 512 tokens × 3 layer hidden states (each 4096-dim, BF16):
> - Cache size: 30 000 × 3 × 4096 × 2 bytes = ~0.74 GB — easily fits on NVMe and in RAM.
> - Pre-extraction time (one pass over 30 K sequences at batch 16): ~3 h.
> - With caching, drafter-only training time: 11 500 × 0.3 s = **~1 h**.

**Recommended approach**: pre-extract and cache target features, then train drafter independently. Total: ~4 h extraction + training.

### 6.4 Evaluation Time Estimate

| Eval | Estimated time |
|---|---|
| GSM8K (1319, greedy, batch 1) | ~30 min per system × 4 systems = ~2 h |
| HumanEval (164, batch 1) | ~15 min per system × 4 = ~1 h |
| MT-Bench (80 multi-turn) | ~20 min per system × 4 = ~1.5 h |
| UltraChat held-out (2000 samples) | ~30 min per system × 4 = ~2 h |
| **Total evaluation** | **~6.5 h** |

### 6.5 48-Hour Timeline

| Phase | Task | Estimated duration |
|---|---|---|
| 0 | Environment setup, model downloads | 2 h |
| 1 | Pre-extract Qwen3-8B features for 30 K samples | 3 h |
| 2 | Experiment 1: BD-EAGLE WSD training (primary) | 1 h (with cached features) |
| 3 | Experiment 2: Cold-start ablation training | 1 h |
| 4 | Experiment 3: Anchor masking ablation | 1 h |
| 5 | Experiment 4: Block size sensitivity (eval only) | 0.5 h |
| 6 | Full evaluation suite (4 systems × 4 benchmarks) | 6.5 h |
| 7 | Results analysis and plotting | 2 h |
| **Total** | | **~17 h** |

**Buffer remaining**: ~31 hours — ample time for debugging, re-runs, and extended training if needed.

---

## 7. Implementation Plan

### 7.1 Repository Structure

```
spec-dllm/
├── scripts/
│   ├── extract_features.py        # Pre-extract Qwen3-8B hidden states → cache
│   ├── train_bd_eagle.py          # WSD training loop
│   ├── eval_throughput.py         # Token/s + τ̄ measurement
│   └── eval_quality.py            # GSM8K / HumanEval / MT-Bench / ROUGE
├── bd_eagle/
│   ├── model.py                   # BD-EAGLE drafter (EAGLE-3 + block-causal mask)
│   ├── attention.py               # Block-causal attention mask implementation
│   ├── dataset.py                 # UltraChat loader + masking sampler
│   └── wsd_scheduler.py           # WSD block-size + LR schedule
├── configs/
│   ├── primary.yaml               # Experiment 1 config
│   ├── cold_start.yaml            # Experiment 2 config
│   └── anchor_masking.yaml        # Experiment 3 config
└── RESEARCH_DESIGN.md
```

### 7.2 Key Implementation Details

**Block-causal attention mask (attention.py):**

```python
def block_causal_mask(seq_len: int, block_size: int) -> torch.Tensor:
    # Returns additive mask: 0 = attend, -inf = ignore
    pos = torch.arange(seq_len)
    block_id = pos // block_size
    # attend if same block OR earlier block
    mask = (block_id.unsqueeze(0) > block_id.unsqueeze(1)).float() * -1e9
    return mask  # shape [seq_len, seq_len]
```

At `block_size = 1`, this reduces to the standard lower-triangular causal mask.

**WSD scheduler (wsd_scheduler.py):**

```python
@dataclass
class WSDConfig:
    warmup_sizes: list = (1, 2, 4, 8)   # block sizes during warmup
    stable_size: int = 8
    decay_sizes: list = (4, 2)
    steps_per_warmup_size: int = 500
    stable_steps: int = 8000
    steps_per_decay_size: int = 500
    lr_max: float = 3e-4
    lr_min: float = 3e-5
```

**Masked diffusion loss:**

```python
def bd_loss(logits, targets, mask_positions, t):
    # logits: [B, L, V], targets: [B, L], mask_positions: [B, L] bool
    weight = -torch.autograd.functional._grad_of_alpha(t)  # α'_t / (1 - α_t)
    ce = F.cross_entropy(logits[mask_positions], targets[mask_positions], reduction='none')
    return (weight * ce).mean()
```

For simplicity, use the noise schedule `α_t = 1 - t` (linear schedule, equivalent to uniform masking rate `t`), so `α'_t / (1 - α_t) = 1/t`. In practice, clip `t` away from 0 to avoid numerical instability.

**Training loop (train_bd_eagle.py):**

At each step:
1. Load pre-cached target features for batch.
2. Sample `t ~ Uniform(0.05, 1.0)`.
3. Mask tokens in current block with probability `t`.
4. Run drafter forward with block-causal mask for current block size.
5. Compute BD loss over masked positions only.
6. Backward + AdamW step.
7. Every 500 steps: update block size per WSD schedule, update LR.

### 7.3 Evaluation Protocol

**Throughput measurement (eval_throughput.py):**
- Generate 500 sequences from GSM8K prompts.
- Measure wall-clock time from first input token to last output token.
- Report median tokens/second and τ̄ (mean tokens accepted per verification step).
- Warm up for 10 sequences before timing.

**Quality check:**
- All systems must match standard greedy decoding on GSM8K exact-match within 0.5% (lossless verification guarantee).
- If any quality degradation > 0.5% is observed, report it as a finding rather than filtering it.

---

## 8. Expected Results and Success Criteria

| Metric | EAGLE-3 baseline | DFlash (from paper) | BD-EAGLE target | Minimum success |
|---|---|---|---|---|
| τ̄ (mean accepted tokens) | ~3.5 | ~8–10 | ~5–7 | > EAGLE-3 |
| Tokens/second | ~40–50 | ~120–150 | ~80–110 | > EAGLE-3 × 1.5 |
| GSM8K accuracy | Matches greedy | Matches greedy | Matches greedy | Δ < 0.5% |
| Training samples | 532 K | 800 K | 30 K | — |
| Training time | ~hours (pre-trained) | ~days | < 5 h | < 24 h |

**Minimum viable result**: BD-EAGLE outperforms EAGLE-3 in tokens/second by ≥ 1.5× with no quality degradation, using ≤ 30 K training samples.

**Strong result**: BD-EAGLE reaches ≥ 70% of DFlash's τ̄ while using < 4% of DFlash's training data.

---

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| EAGLE-3 single layer too shallow for block diffusion — acceptance rate does not improve over AR drafter | Medium | Add 2 additional randomly-initialised Transformer layers on top of the EAGLE-3 layer; run as Experiment 5 if time allows |
| WSD warmup destabilises EAGLE-3 weights at block_size transition | Low–Medium | Use smaller LR (1×10⁻⁴) in first 2 warmup sizes; apply top-k checkpoint merge as in LLaDA 2.0 |
| Memory bandwidth bottleneck limits throughput gains (GB10: 273 GB/s vs A100: 2 TB/s) | Expected | The block diffusion advantage is hardware-agnostic for the drafter; the target model throughput will be the same. Report relative speedup, not absolute tokens/s. |
| `AngelSlim/Qwen3-8B_eagle3` checkpoint not compatible with current EAGLE codebase | Low | Fall back to training EAGLE-3 from scratch with the SafeAILab codebase (adds ~4 h to schedule) |
| 30 K samples insufficient for convergence | Medium | Extend to 100 K samples (still < 800 K DFlash); add ~3 h total training time |

---

## 10. References

1. **EAGLE-3**: Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test," NeurIPS 2025. arXiv:2503.01840.
2. **DFlash**: Chen et al., "DFlash: Block Diffusion for Flash Speculative Decoding," arXiv:2602.06036.
3. **LLaDA 2.0**: InclusionAI, "LLaDA2.0: Scaling Up Diffusion Language Models to 100B," arXiv:2512.15745.
4. **Block Diffusion (BD-LM)**: Arriola et al., "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models," arXiv:2503.09573.
5. **MDLM**: Sahoo et al., "Simple and Effective Masked Diffusion Language Models," NeurIPS 2024. arXiv:2406.07524.
6. **Fast-dLLM v2**: arXiv:2509.26328.
7. **EAGLE GitHub**: https://github.com/SafeAILab/EAGLE
8. **DFlash GitHub**: https://github.com/z-lab/dflash
