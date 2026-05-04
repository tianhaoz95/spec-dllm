"""
Pre-extract frozen target model hidden states and cache them to disk.

This separates the expensive target-model forward pass (Qwen3-8B) from the
cheap drafter training. Cached features are stored as memory-mapped tensors
so they can be loaded lazily during training without duplicating RAM.

Usage:
    python scripts/extract_features.py --config configs/primary.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))
from bd_eagle.dataset import UltraChatDataset, build_dataloader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-extract Qwen3-8B hidden states")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    target_model_id = cfg["target_model"]
    layer_indices: list[int] = cfg.get("target_layer_indices", [])
    max_samples: int = cfg["max_samples"]
    max_length: int = cfg["max_length"]
    cache_dir: str = cfg["feature_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    # Check if cache already exists
    meta_path = Path(cache_dir) / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("model") == target_model_id
            and meta.get("n_samples") == max_samples
            and meta.get("max_length") == max_length
        ):
            print(f"[extract_features] Cache already exists at {cache_dir} — skipping.")
            return

    print(f"[extract_features] Loading tokenizer: {target_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[extract_features] Loading target model: {target_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        output_hidden_states=True,
    )
    model.eval()

    # Determine layer indices if not specified
    num_layers = model.config.num_hidden_layers
    if not layer_indices:
        lo = num_layers // 6
        mid = num_layers // 2
        hi = 5 * num_layers // 6
        layer_indices = [lo, mid, hi]
    print(f"[extract_features] Extracting layers {layer_indices} out of {num_layers}")

    hidden_size = model.config.hidden_size
    n_extract_layers = len(layer_indices)

    print(f"[extract_features] Building dataset ({max_samples} samples, max_length={max_length})")
    dataset = UltraChatDataset(
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_length=max_length,
    )
    n_actual = len(dataset)
    print(f"[extract_features] Dataset size after filtering: {n_actual}")

    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        pad_id=tokenizer.pad_token_id,
        max_length=max_length,
        shuffle=False,
        num_workers=2,
    )

    # Allocate memory-mapped arrays
    feat_path = Path(cache_dir) / "features.npy"
    ids_path = Path(cache_dir) / "input_ids.npy"
    mask_path = Path(cache_dir) / "attention_mask.npy"

    n_batches = len(dataloader)
    n_stored = n_batches * args.batch_size  # may be slightly more than n_actual

    features_mmap = np.lib.format.open_memmap(
        feat_path,
        mode="w+",
        dtype="float16",
        shape=(n_stored, max_length, n_extract_layers * hidden_size),
    )
    ids_mmap = np.lib.format.open_memmap(
        ids_path, mode="w+", dtype="int32", shape=(n_stored, max_length)
    )
    mask_mmap = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype="uint8", shape=(n_stored, max_length)
    )

    write_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            # hidden_states: tuple of (num_layers+1) tensors [B, T, H]
            # index 0 = embedding layer; +1 offset for actual transformer layers
            hs = out.hidden_states
            extracted = torch.stack(
                [hs[i + 1] for i in layer_indices], dim=2
            )  # [B, T, n_layers, H]
            # reshape to [B, T, n_layers * H]
            B, T, NL, H = extracted.shape
            fused = extracted.reshape(B, T, NL * H).cpu().to(torch.float16).numpy()

            batch_size_actual = B
            end_idx = write_idx + batch_size_actual
            features_mmap[write_idx:end_idx] = fused
            ids_mmap[write_idx:end_idx] = input_ids.cpu().numpy().astype("int32")
            mask_mmap[write_idx:end_idx] = attention_mask.cpu().numpy().astype("uint8")
            write_idx = end_idx

    # Trim to actual written rows
    # (np memmap can't be shrunk in-place, so record actual count in meta)
    meta = {
        "model": target_model_id,
        "n_samples": n_actual,
        "n_stored": write_idx,
        "max_length": max_length,
        "layer_indices": layer_indices,
        "hidden_size": hidden_size,
        "n_extract_layers": n_extract_layers,
        "fused_dim": n_extract_layers * hidden_size,
        "dtype": "float16",
        "feat_path": str(feat_path),
        "ids_path": str(ids_path),
        "mask_path": str(mask_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[extract_features] Done. Wrote {write_idx} samples to {cache_dir}")
    print(f"  Features shape: ({write_idx}, {max_length}, {n_extract_layers * hidden_size})")
    est_gb = (write_idx * max_length * n_extract_layers * hidden_size * 2) / 1e9
    print(f"  Estimated disk size: {est_gb:.2f} GB")


if __name__ == "__main__":
    main()
