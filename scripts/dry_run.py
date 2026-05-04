"""
Dry-run validation script for BD-EAGLE.

Exercises every component with tiny dummy data to confirm correctness and GPU
usage without loading real model weights or datasets. Completes in seconds.

Usage:
    python scripts/dry_run.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

# Make sure the project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from bd_eagle.attention import block_causal_mask
from bd_eagle.dataset import sample_block_masks
from bd_eagle.model import BDEagleDrafter
from bd_eagle.wsd_scheduler import WSDConfig, WSDScheduler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tiny model dimensions — fast to instantiate and run
H   = 64      # hidden_size
I   = 128     # intermediate_size
NQH = 4       # num_q_heads
NKH = 2       # num_kv_heads
HD  = 16      # head_dim
FV  = 200     # full_vocab_size
DV  = 100     # draft_vocab_size

B   = 2       # batch size
T   = 16      # sequence length
EOS = 150     # eos / mask token id (must be in [DV, FV) to be outside draft vocab)

N_STEPS = 5   # training steps to run

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_failed: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    if not ok:
        _failed.append(name)


# ---------------------------------------------------------------------------
# Helper: build a tiny model with synthetic vocab mappings
# ---------------------------------------------------------------------------

def make_model() -> BDEagleDrafter:
    model = BDEagleDrafter(
        hidden_size=H,
        intermediate_size=I,
        num_q_heads=NQH,
        num_kv_heads=NKH,
        head_dim=HD,
        full_vocab_size=FV,
        draft_vocab_size=DV,
        block_size=1,
    )
    # Populate vocab-mapping buffers:
    # First DV target tokens map into the draft vocab; the rest do not.
    t2d = torch.zeros(FV, dtype=torch.bool)
    t2d[:DV] = True
    d2t = torch.arange(DV, dtype=torch.long)
    model.t2d.copy_(t2d)
    model.d2t.copy_(d2t)
    model.embed_tokens.weight.requires_grad_(False)
    return model.to(DEVICE)


def make_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (input_ids, attention_mask, fused_features) on DEVICE."""
    # Mix of draft-vocab tokens (0..DV-1) and out-of-draft tokens (DV..FV-1)
    # so the loss filter is exercised on both sides.
    input_ids = torch.randint(0, FV, (B, T), device=DEVICE)
    attention_mask = torch.ones(B, T, dtype=torch.long, device=DEVICE)
    # Last few positions are padding for one sample
    attention_mask[0, T - 3:] = 0
    fused_features = torch.randn(B, T, 3 * H, device=DEVICE)
    return input_ids, attention_mask, fused_features


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_gpu() -> None:
    print("\n[GPU]")
    check("CUDA available", torch.cuda.is_available(), DEVICE.type)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        check("Device name non-empty", bool(name), name)


def test_block_causal_mask() -> None:
    print("\n[block_causal_mask]")
    seq = 8
    for bs in (1, 2, 4):
        mask = block_causal_mask(seq, bs, DEVICE)
        check(f"shape bs={bs}", mask.shape == (seq, seq), str(mask.shape))
        check(f"only 0 or -1e9 bs={bs}", ((mask == 0.0) | (mask == -1e9)).all().item())
        # Diagonal must be 0 (self-attention always allowed)
        check(f"diagonal zero bs={bs}", (mask.diagonal() == 0.0).all().item())
        # Position 0 attends to its own block only; all later blocks are blocked
        check(f"causal property bs={bs}", (mask[0, bs:] == -1e9).all().item())
        # Within-block: position 1 can attend to position 0 if bs >= 2
        if bs >= 2:
            check(f"within-block attend bs={bs}", mask[1, 0].item() == 0.0)


def test_sample_block_masks() -> None:
    print("\n[sample_block_masks]")
    input_ids = torch.randint(0, FV, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[0, T - 3:] = 0

    mask_pos, t = sample_block_masks(input_ids, attention_mask, block_size=4)
    check("mask_positions shape", mask_pos.shape == (B, T), str(mask_pos.shape))
    check("mask_positions dtype bool", mask_pos.dtype == torch.bool)
    check("t in [0.05, 1.0]", 0.05 <= t <= 1.0, f"t={t:.4f}")
    # Padding positions must NOT be masked
    check("padding not masked", not mask_pos[0, T - 3:].any().item())


def test_model_forward() -> None:
    print("\n[BDEagleDrafter forward]")
    model = make_model()
    input_ids, attention_mask, fused_features = make_batch()

    with torch.no_grad():
        for bs in (1, 2, 4):
            logits = model(input_ids, fused_features, block_size=bs)
            check(f"output shape bs={bs}", logits.shape == (B, T, DV), str(logits.shape))
            check(f"output on GPU bs={bs}", logits.device.type == "cuda")
            check(f"output finite bs={bs}", logits.isfinite().all().item())


def test_loss() -> None:
    print("\n[masked_diffusion_loss]")
    model = make_model()
    input_ids, attention_mask, fused_features = make_batch()
    mask_pos, t = sample_block_masks(
        input_ids.cpu(), attention_mask.cpu(), block_size=4
    )
    mask_pos = mask_pos.to(DEVICE)

    logits = model(input_ids, fused_features, block_size=4)
    loss = BDEagleDrafter.masked_diffusion_loss(
        logits, input_ids, mask_pos, model.t2d, t, block_size=4
    )
    check("loss is scalar", loss.shape == torch.Size([]))
    check("loss is finite", loss.isfinite().item())
    check("loss >= 0", loss.item() >= 0, f"loss={loss.item():.4f}")

    # With position weighting
    loss_pw = BDEagleDrafter.masked_diffusion_loss(
        logits, input_ids, mask_pos, model.t2d, t,
        use_position_weight=True, block_size=4, gamma=5.0
    )
    check("position-weighted loss finite", loss_pw.isfinite().item())


def test_backward() -> None:
    print("\n[backward / gradient flow]")
    model = make_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    check("trainable params exist", len(trainable) > 0, f"{len(trainable)} params")

    input_ids, attention_mask, fused_features = make_batch()
    mask_pos, t = sample_block_masks(
        input_ids.cpu(), attention_mask.cpu(), block_size=2
    )
    mask_pos = mask_pos.to(DEVICE)

    logits = model(input_ids, fused_features, block_size=2)
    loss = BDEagleDrafter.masked_diffusion_loss(
        logits, input_ids, mask_pos, model.t2d, t, block_size=2
    )

    # Handle zero-loss edge case (no active positions)
    if loss.item() == 0.0:
        loss = logits.sum() * 0.0 + torch.tensor(1.0, requires_grad=True, device=DEVICE)
        loss = loss.mean()

    loss.backward()
    grads_exist = any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    check("gradients non-zero", grads_exist)

    check("embed_tokens frozen", model.embed_tokens.weight.grad is None)


def test_wsd_scheduler() -> None:
    print("\n[WSDScheduler]")
    cfg = WSDConfig(
        warmup_sizes=[1, 2],
        stable_size=4,
        decay_sizes=[2],
        steps_per_warmup_size=3,
        stable_steps=4,
        steps_per_decay_size=2,
        lr_max=3e-4,
        lr_min=3e-5,
    )
    sched = WSDScheduler(cfg)
    total = (2 * 3) + 4 + (1 * 2)  # = 12
    check("total_steps", sched.total_steps == total, f"{sched.total_steps}")

    # Warmup step 0: block_size=1, lr << lr_max
    bs0, lr0 = sched[0]
    check("warmup block_size=1", bs0 == 1, f"bs={bs0}")
    check("warmup lr < lr_max", lr0 < cfg.lr_max, f"lr={lr0:.2e}")

    # Stable phase starts at step 6 (2 warmup sizes × 3 steps)
    bs_s, lr_s = sched[6]
    check("stable block_size=4", bs_s == cfg.stable_size, f"bs={bs_s}")
    check("stable lr == lr_max", abs(lr_s - cfg.lr_max) < 1e-9, f"lr={lr_s:.2e}")

    # Decay: last step has lr close to lr_min
    bs_d, lr_d = sched[total - 1]
    check("decay block_size=2", bs_d == 2, f"bs={bs_d}")
    check("decay lr < lr_max", lr_d < cfg.lr_max, f"lr={lr_d:.2e}")


def test_training_loop() -> None:
    print(f"\n[end-to-end training loop — {N_STEPS} steps]")
    model = make_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    cfg = WSDConfig(
        warmup_sizes=[1, 2, 4],
        stable_size=4,
        decay_sizes=[2],
        steps_per_warmup_size=2,
        stable_steps=2,
        steps_per_decay_size=1,
        lr_max=3e-4,
        lr_min=3e-5,
    )
    sched = WSDScheduler(cfg)

    losses = []
    t0 = time.perf_counter()

    for step in range(N_STEPS):
        block_size, lr = sched[step]
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        input_ids, attention_mask, fused_features = make_batch()
        mask_pos, t = sample_block_masks(
            input_ids.cpu(), attention_mask.cpu(), block_size=block_size
        )
        mask_pos = mask_pos.to(DEVICE)

        # Apply mask: replace masked positions with eos id
        masked_ids = input_ids.clone()
        masked_ids[mask_pos] = EOS

        logits = model(masked_ids, fused_features, block_size=block_size)
        loss = BDEagleDrafter.masked_diffusion_loss(
            logits, input_ids, mask_pos, model.t2d, t, block_size=block_size
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        optimizer.step()

        losses.append(loss.item())
        print(f"    step {step:2d}  bs={block_size}  lr={lr:.2e}  loss={loss.item():.4f}")

    elapsed = time.perf_counter() - t0
    all_finite = all(torch.isfinite(torch.tensor(l)).item() for l in losses)
    check("all losses finite", all_finite)
    check(f"completed {N_STEPS} steps in {elapsed:.1f}s", True, f"{elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("BD-EAGLE dry-run validation")
    print(f"Device: {DEVICE}  |  Model dims: H={H}, T={T}, B={B}")
    print("=" * 60)

    test_gpu()
    test_block_causal_mask()
    test_sample_block_masks()
    test_model_forward()
    test_loss()
    test_backward()
    test_wsd_scheduler()
    test_training_loop()

    print("\n" + "=" * 60)
    if _failed:
        print(f"\033[31mFAILED: {len(_failed)} check(s)\033[0m")
        for name in _failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"\033[32mAll checks passed.\033[0m")
