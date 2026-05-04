"""
Dataset and masking utilities for BD-EAGLE training.
"""

from __future__ import annotations

import random
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


MASK_TOKEN_PLACEHOLDER = -1  # placeholder before actual mask-token id is known


class UltraChatDataset(Dataset):
    """
    Loads UltraChat-200K from HuggingFace datasets and tokenizes
    assistant responses for training the drafter.

    For each sample we return the full conversation tokenized up to max_length.
    The training loop will decide which positions to mask.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train_sft",
        max_samples: int = 30_000,
        max_length: int = 512,
        seed: int = 42,
    ):
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
        if max_samples < len(ds):
            ds = ds.shuffle(seed=seed).select(range(max_samples))

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for item in ds:
            messages = item["messages"]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            ids = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )["input_ids"]
            if len(ids) > 4:  # skip degenerate samples
                self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> list[int]:
        return self.samples[idx]


def collate_fn(batch: list[list[int]], pad_id: int, max_length: int) -> dict[str, Tensor]:
    """Pad a batch of token id lists to the same length."""
    max_len = min(max(len(s) for s in batch), max_length)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, ids in enumerate(batch):
        length = min(len(ids), max_len)
        input_ids[i, :length] = torch.tensor(ids[:length], dtype=torch.long)
        attention_mask[i, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def sample_block_masks(
    input_ids: Tensor,
    attention_mask: Tensor,
    block_size: int,
    t_min: float = 0.05,
    t_max: float = 1.0,
    n_anchors_per_block: int = 0,
) -> tuple[Tensor, Tensor, float]:
    """
    Sample a masking pattern for the current batch.

    Returns:
        masked_ids:     [B, T] token ids with masked positions set to mask_token_id
        mask_positions: [B, T] bool, True = position was masked
        t:              scalar noise level used
    """
    t = random.uniform(t_min, t_max)
    B, T = input_ids.shape
    mask_positions = torch.zeros(B, T, dtype=torch.bool)

    # Only mask valid (non-padding) positions
    valid = attention_mask.bool()

    for b in range(B):
        for block_start in range(0, T, block_size):
            block_end = min(block_start + block_size, T)
            block_valid = valid[b, block_start:block_end]
            block_len = block_valid.sum().item()
            if block_len == 0:
                continue

            valid_positions = block_valid.nonzero(as_tuple=True)[0] + block_start

            # Sample anchor positions (always kept unmasked)
            anchors = set()
            if n_anchors_per_block > 0 and len(valid_positions) > n_anchors_per_block:
                anchor_idxs = random.sample(range(len(valid_positions)), n_anchors_per_block)
                anchors = {valid_positions[i].item() for i in anchor_idxs}

            # Mask each non-anchor valid position with probability t
            for pos in valid_positions.tolist():
                if pos not in anchors and random.random() < t:
                    mask_positions[b, pos] = True

    return mask_positions, t


def build_dataloader(
    dataset: UltraChatDataset,
    batch_size: int,
    pad_id: int,
    max_length: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id, max_length=max_length),
        pin_memory=True,
        drop_last=True,
    )
