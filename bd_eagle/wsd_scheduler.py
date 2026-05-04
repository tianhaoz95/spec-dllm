"""
Warmup-Stable-Decay scheduler for block size and learning rate.

Mirrors the LLaDA 2.0 WSD curriculum, scaled down to speculative-decoding
block sizes (max=8 instead of 4096).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class WSDConfig:
    # Block size trajectory
    warmup_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    stable_size: int = 8
    decay_sizes: list[int] = field(default_factory=lambda: [4, 2])

    # Steps at each stage
    steps_per_warmup_size: int = 500
    stable_steps: int = 8000
    steps_per_decay_size: int = 500

    # Learning rate
    lr_max: float = 3e-4
    lr_min: float = 3e-5


class WSDScheduler:
    """
    Yields (block_size, lr) at each training step.

    Phases:
      Warmup:  steps_per_warmup_size steps at each block size in warmup_sizes
      Stable:  stable_steps at stable_size
      Decay:   steps_per_decay_size at each size in decay_sizes, lr cosine-decays
    """

    def __init__(self, cfg: WSDConfig):
        self.cfg = cfg
        self._build_schedule()

    def _build_schedule(self) -> None:
        c = self.cfg
        schedule = []  # list of (block_size, phase_name, local_step, phase_total)

        # Warmup phase: LR linearly warms up from 0 to lr_max
        warmup_total = len(c.warmup_sizes) * c.steps_per_warmup_size
        for i, bs in enumerate(c.warmup_sizes):
            for s in range(c.steps_per_warmup_size):
                global_warmup_step = i * c.steps_per_warmup_size + s
                lr = c.lr_max * (global_warmup_step + 1) / warmup_total
                schedule.append((bs, lr))

        # Stable phase: constant LR
        for _ in range(c.stable_steps):
            schedule.append((c.stable_size, c.lr_max))

        # Decay phase: LR cosine-decays from lr_max to lr_min
        decay_total = len(c.decay_sizes) * c.steps_per_decay_size
        for i, bs in enumerate(c.decay_sizes):
            for s in range(c.steps_per_decay_size):
                global_decay_step = i * c.steps_per_decay_size + s
                cos_val = math.cos(math.pi * global_decay_step / decay_total)
                lr = c.lr_min + 0.5 * (c.lr_max - c.lr_min) * (1 + cos_val)
                schedule.append((bs, lr))

        self.schedule = schedule

    @property
    def total_steps(self) -> int:
        return len(self.schedule)

    def __getitem__(self, step: int) -> tuple[int, float]:
        """Return (block_size, lr) for the given step index."""
        if step >= len(self.schedule):
            # After schedule ends, hold the last values
            return self.schedule[-1]
        return self.schedule[step]

    def __len__(self) -> int:
        return self.total_steps
