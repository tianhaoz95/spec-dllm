"""
WSD training loop for BD-EAGLE.

Reads pre-extracted feature cache (run extract_features.py first).
Trains only the EAGLE-3 drafter weights using the BD-LM objective.

Usage:
    python scripts/train_bd_eagle.py --config configs/primary.yaml
    python scripts/train_bd_eagle.py --config configs/cold_start.yaml
    python scripts/train_bd_eagle.py --config configs/anchor_masking.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from bd_eagle.model import BDEagleDrafter
from bd_eagle.wsd_scheduler import WSDConfig, WSDScheduler
from bd_eagle.cached_dataset import build_cached_dataloader
from bd_eagle.dataset import sample_block_masks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None, help="Checkpoint dir to resume from")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(step: int, model: BDEagleDrafter, optimizer: AdamW, cfg: dict, out_dir: str) -> None:
    ckpt_dir = Path(out_dir) / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save trainable weights only (not frozen embed_tokens)
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("embed_tokens")
    }
    torch.save(trainable_state, ckpt_dir / "drafter_weights.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    (ckpt_dir / "train_state.json").write_text(
        json.dumps({"step": step, "config": cfg}, indent=2)
    )
    print(f"[train] Checkpoint → {ckpt_dir}")

    # Keep only the last 3 checkpoints
    existing = sorted(Path(out_dir).glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    for old in existing[:-3]:
        shutil.rmtree(old)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    run_name: str = cfg.get("run_name", "bd_eagle")
    out_dir: str = cfg.get("output_dir", f"checkpoints/{run_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Copy config to output dir for reproducibility
    import shutil as _sh
    _sh.copy2(args.config, Path(out_dir) / "config.yaml")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    eagle_ckpt = cfg["eagle_model"]
    # If it's a HuggingFace model ID, resolve the local cache path
    if not Path(eagle_ckpt).exists():
        from huggingface_hub import snapshot_download
        eagle_ckpt = snapshot_download(eagle_ckpt)

    print(f"[train] Loading BD-EAGLE drafter from: {eagle_ckpt}")
    model = BDEagleDrafter.from_pretrained(eagle_ckpt, device=device)

    if cfg.get("cold_start", False):
        print("[train] Cold start: re-initialising trainable weights randomly")
        for name, param in model.named_parameters():
            if "embed_tokens" not in name and param.requires_grad:
                if param.dim() >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)

    # Freeze embed_tokens (shared with target, not updated)
    model.embed_tokens.weight.requires_grad_(False)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[train] Trainable parameters: {n_trainable / 1e6:.1f} M")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print(f"[train] Loading feature cache: {cfg['feature_cache_dir']}")
    dataloader = build_cached_dataloader(
        cache_dir=cfg["feature_cache_dir"],
        batch_size=cfg["batch_size"],
        n_samples=cfg.get("max_samples"),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
    )

    # ------------------------------------------------------------------
    # WSD scheduler
    # ------------------------------------------------------------------
    wsd_cfg = WSDConfig(
        warmup_sizes=cfg.get("warmup_sizes", [1, 2, 4, 8]),
        stable_size=cfg.get("stable_size", 8),
        decay_sizes=cfg.get("decay_sizes", [4, 2]),
        steps_per_warmup_size=cfg.get("steps_per_warmup_size", 500),
        stable_steps=cfg.get("stable_steps", 8000),
        steps_per_decay_size=cfg.get("steps_per_decay_size", 500),
        lr_max=cfg.get("lr_max", 3e-4),
        lr_min=cfg.get("lr_min", 3e-5),
    )
    scheduler = WSDScheduler(wsd_cfg)
    print(f"[train] WSD total steps: {scheduler.total_steps}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = AdamW(
        trainable_params,
        lr=wsd_cfg.lr_max,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )

    use_anchor = cfg.get("use_anchor_masking", False)
    n_anchors  = cfg.get("n_anchors_per_block", 0)
    use_pw     = cfg.get("use_position_weight", False)
    gamma      = cfg.get("loss_gamma", 5.0)
    grad_clip  = cfg.get("grad_clip", 0.5)
    log_every  = cfg.get("log_every", 50)
    save_every = cfg.get("save_every", 500)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_step = 0
    if args.resume:
        resume_dir = Path(args.resume)
        state_dict = torch.load(resume_dir / "drafter_weights.pt", map_location=device)
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(torch.load(resume_dir / "optimizer.pt", map_location=device))
        train_state = json.loads((resume_dir / "train_state.json").read_text())
        start_step = train_state["step"]
        print(f"[train] Resumed from step {start_step}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_path = Path(out_dir) / "train_log.jsonl"
    log_file = open(log_path, "a")

    data_iter = iter(dataloader)
    t_start = time.time()
    model.train()

    pbar = tqdm(
        range(start_step, scheduler.total_steps),
        initial=start_step,
        total=scheduler.total_steps,
        desc=run_name,
    )

    for step in pbar:
        block_size, lr = scheduler[step]
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Fetch batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        fused_features = batch["fused_features"].to(device, dtype=torch.bfloat16)

        # Sample masking pattern
        mask_positions, t_noise = sample_block_masks(
            input_ids=input_ids.cpu(),
            attention_mask=attention_mask.cpu(),
            block_size=block_size,
            n_anchors_per_block=n_anchors if use_anchor else 0,
        )
        mask_positions = mask_positions.to(device)

        # Replace masked positions with eos token id (used as mask placeholder)
        eos_id = 151645  # Qwen3 eos token id
        masked_ids = input_ids.clone()
        masked_ids[mask_positions] = eos_id

        # Forward
        optimizer.zero_grad()
        logits = model(masked_ids, fused_features, block_size=block_size)

        # BD-LM loss
        loss = BDEagleDrafter.masked_diffusion_loss(
            logits=logits,
            targets=input_ids,
            mask_positions=mask_positions,
            t2d=model.t2d,
            t=t_noise,
            use_position_weight=use_pw,
            block_size=block_size,
            gamma=gamma,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()

        loss_val = loss.item()

        if step % log_every == 0:
            elapsed = time.time() - t_start
            record = {
                "step": step,
                "loss": loss_val,
                "block_size": block_size,
                "lr": lr,
                "t_noise": t_noise,
                "elapsed_s": elapsed,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            pbar.set_postfix(loss=f"{loss_val:.4f}", bs=block_size, lr=f"{lr:.2e}")

        if step % save_every == 0 and step > 0:
            save_checkpoint(step, model, optimizer, cfg, out_dir)

    save_checkpoint(step, model, optimizer, cfg, out_dir)
    log_file.close()
    print(f"[train] Done. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
