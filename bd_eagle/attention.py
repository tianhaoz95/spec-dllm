import torch
from torch import Tensor


def block_causal_mask(seq_len: int, block_size: int, device: torch.device) -> Tensor:
    """
    Block-causal attention mask.

    Within the same block: all positions attend to each other (bidirectional).
    Across blocks: later blocks cannot attend to future blocks (causal).

    Returns additive mask: 0 = attend, -1e9 = ignore.
    Shape: [seq_len, seq_len]
    """
    pos = torch.arange(seq_len, device=device)
    block_id = pos // block_size

    # position i attends to position j iff block_id[j] <= block_id[i]
    can_attend = block_id.unsqueeze(1) >= block_id.unsqueeze(0)  # [seq, seq]
    mask = torch.where(can_attend, torch.zeros(1, device=device), torch.full((1,), -1e9, device=device))
    return mask  # [seq_len, seq_len]


def causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Standard lower-triangular causal mask (block_size=1 special case)."""
    return block_causal_mask(seq_len, block_size=1, device=device)
