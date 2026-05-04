"""
Dataset that reads from pre-extracted feature cache (memory-mapped numpy arrays).
Used during drafter training to avoid re-running the target model every step.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from .dataset import sample_block_masks


class CachedFeatureDataset(Dataset):
    """
    Reads pre-extracted features, input_ids, and attention_mask from disk.
    Memory-mapped so the full array does not need to fit in RAM simultaneously.
    """

    def __init__(self, cache_dir: str, n_samples: int | None = None):
        meta_path = Path(cache_dir) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Feature cache not found at {cache_dir}. "
                "Run scripts/extract_features.py first."
            )
        self.meta = json.loads(meta_path.read_text())
        n = self.meta["n_stored"]
        if n_samples is not None:
            n = min(n, n_samples)
        self.n = n

        self.features = np.load(self.meta["feat_path"], mmap_mode="r")
        self.input_ids = np.load(self.meta["ids_path"], mmap_mode="r")
        self.attention_mask = np.load(self.meta["mask_path"], mmap_mode="r")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx].astype("int64"), dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx].astype("int64"), dtype=torch.long),
            "fused_features": torch.tensor(self.features[idx].astype("float32"), dtype=torch.float),
        }

    @property
    def fused_dim(self) -> int:
        return self.meta["fused_dim"]


def cached_collate_fn(batch: list[dict]) -> dict[str, Tensor]:
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    fused_features = torch.stack([b["fused_features"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "fused_features": fused_features,
    }


def build_cached_dataloader(
    cache_dir: str,
    batch_size: int,
    n_samples: int | None = None,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    ds = CachedFeatureDataset(cache_dir, n_samples=n_samples)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=cached_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
