"""
BD-EAGLE drafter: self-contained EAGLE-3 implementation with block-causal attention.

Architecture (Qwen3-8B target, from checkpoint inspection):
  - fc:       Linear(hidden_size * 3, hidden_size, bias=False)
  - midlayer: single Transformer decoder layer
      - hidden_norm:               RMSNorm(hidden_size)
      - input_layernorm:           RMSNorm(hidden_size)
      - self_attn (GQA, input_dim=2*hidden_size):
          q_proj: Linear(2H, H, bias=False)
          k_proj: Linear(2H, num_kv_heads*head_dim, bias=False)
          v_proj: Linear(2H, num_kv_heads*head_dim, bias=False)
          o_proj: Linear(H, H, bias=False)
      - post_attention_layernorm: RMSNorm(hidden_size)
      - mlp (SwiGLU):
          gate_proj: Linear(H, I, bias=False)
          up_proj:   Linear(H, I, bias=False)
          down_proj: Linear(I, H, bias=False)
  - norm:    RMSNorm(hidden_size)
  - lm_head: Linear(hidden_size, draft_vocab_size, bias=False)
  - embed_tokens: Embedding(full_vocab_size, hidden_size)  [shared from target, frozen]
  - d2t: [draft_vocab_size] int64  — maps draft → target token id
  - t2d: [full_vocab_size]  bool   — True if target token in draft vocab

Forward:
  1. projected = fc( cat([hs_low, hs_mid, hs_high], dim=-1) )  [B, T, H]
  2. norm_proj  = hidden_norm(projected)
  3. norm_emb   = input_layernorm(embed_tokens(input_ids))
  4. x          = cat([norm_emb, norm_proj], dim=-1)            [B, T, 2H]
  5. attn_out   = self_attn(x, block_causal_mask)               [B, T, H]
  6. h          = projected + attn_out                          [B, T, H]
  7. h          = h + mlp(post_attention_layernorm(h))
  8. logits     = lm_head(norm(h))                              [B, T, draft_vocab_size]
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import block_causal_mask


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 40960, base: float = 1_000_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self._cos_cached is not None
            and self._cos_cached.shape[0] >= seq_len
            and self._cos_cached.device == device
        ):
            return
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))  # [T, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)            # [T, head_dim]
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        self._build_cache(position_ids.max().item() + 1, x.device, x.dtype)
        cos = self._cos_cached[position_ids]  # [B, T, head_dim]
        sin = self._sin_cached[position_ids]
        return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    # cos/sin: [B, T, head_dim] — unsqueeze for head dimension
    cos = cos.unsqueeze(2)  # [B, T, 1, head_dim]
    sin = sin.unsqueeze(2)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class GQAttention(nn.Module):
    """Grouped-query attention with support for block-causal masks."""

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 1_000_000.0,
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_q_heads // num_kv_heads

        # Input dimension is 2 * hidden_size (EAGLE-3 concatenation)
        input_dim = 2 * hidden_size
        self.q_proj = nn.Linear(input_dim, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(head_dim, base=rope_theta)

    def forward(
        self,
        x: Tensor,                        # [B, T, 2H]
        attention_mask: Tensor | None,    # [1, 1, T, T] additive mask
        position_ids: Tensor,             # [B, T]
    ) -> Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim)   # [B, T, Hq, D]
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)  # [B, T, Hkv, D]
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)  # [B, T, Hkv, D]

        cos, sin = self.rotary(q, position_ids)
        q, k = apply_rotary(q, k, cos, sin)

        # Transpose to [B, heads, T, D] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat K/V for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, Hq, T, T]

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)                           # [B, Hq, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, Hq*D]
        return self.o_proj(out)


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Main drafter
# ---------------------------------------------------------------------------

class BDEagleDrafter(nn.Module):
    """
    EAGLE-3 single-layer drafter adapted for block diffusion.

    Loads weights from the AngelSlim/Qwen3-8B_eagle3 checkpoint.
    The only training-time change is the attention mask: at block_size > 1
    the mask becomes block-causal instead of purely causal.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_q_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        full_vocab_size: int = 151936,
        draft_vocab_size: int = 32000,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        block_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.block_size = block_size

        # Feature projection (3 target layers → 1 hidden)
        self.fc = nn.Linear(hidden_size * 3, hidden_size, bias=False)

        # Norms
        self.hidden_norm      = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_layernorm  = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attn_norm   = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm             = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Attention
        self.self_attn = GQAttention(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
        )

        # MLP
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

        # Vocabulary head and embedding (draft vocab)
        self.embed_tokens = nn.Embedding(full_vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, draft_vocab_size, bias=False)

        # Vocab mapping buffers (populated when checkpoint is loaded)
        self.register_buffer("d2t", torch.zeros(draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(full_vocab_size, dtype=torch.bool))

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        target_embed_tokens: nn.Embedding | None = None,
        block_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> "BDEagleDrafter":
        """
        Load from an AngelSlim-style EAGLE-3 checkpoint.

        Args:
            checkpoint_path: path to directory containing pytorch_model.bin
            target_embed_tokens: the frozen embedding from the target model (shared)
            block_size: initial block size (can be changed later)
            device: target device
        """
        import json

        ckpt_dir = Path(checkpoint_path)
        state_path = ckpt_dir / "pytorch_model.bin"
        config_path = ckpt_dir / "config.json"

        with open(config_path) as f:
            config = json.load(f)

        model = cls(
            hidden_size=config.get("hidden_size", 4096),
            intermediate_size=config.get("intermediate_size", 12288),
            num_q_heads=config.get("num_attention_heads", 32),
            num_kv_heads=config.get("num_key_value_heads", 8),
            head_dim=config.get("head_dim", 128),
            full_vocab_size=config.get("vocab_size", 151936),
            draft_vocab_size=config.get("draft_vocab_size", 32000),
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            rope_theta=config.get("rope_theta", 1_000_000.0),
            block_size=block_size,
        )

        state = torch.load(state_path, map_location="cpu", weights_only=True)

        # Map checkpoint keys → model attribute paths
        key_map = {
            "fc.weight":                                          "fc.weight",
            "midlayer.hidden_norm.weight":                        "hidden_norm.weight",
            "midlayer.input_layernorm.weight":                    "input_layernorm.weight",
            "midlayer.post_attention_layernorm.weight":           "post_attn_norm.weight",
            "midlayer.self_attn.q_proj.weight":                   "self_attn.q_proj.weight",
            "midlayer.self_attn.k_proj.weight":                   "self_attn.k_proj.weight",
            "midlayer.self_attn.v_proj.weight":                   "self_attn.v_proj.weight",
            "midlayer.self_attn.o_proj.weight":                   "self_attn.o_proj.weight",
            "midlayer.mlp.gate_proj.weight":                      "mlp.gate_proj.weight",
            "midlayer.mlp.up_proj.weight":                        "mlp.up_proj.weight",
            "midlayer.mlp.down_proj.weight":                      "mlp.down_proj.weight",
            "norm.weight":                                        "norm.weight",
            "lm_head.weight":                                     "lm_head.weight",
            "d2t":                                                "d2t",
            "t2d":                                                "t2d",
        }

        mapped = {}
        for ckpt_key, model_key in key_map.items():
            if ckpt_key in state:
                mapped[model_key] = state[ckpt_key]
            else:
                print(f"[BDEagleDrafter] WARNING: key {ckpt_key!r} not found in checkpoint")

        model.load_state_dict(mapped, strict=False)

        # Share embedding with target model (frozen)
        if target_embed_tokens is not None:
            model.embed_tokens.weight = target_embed_tokens.weight
            model.embed_tokens.weight.requires_grad_(False)
        else:
            # Freeze our own embedding (will be replaced when target model loads)
            model.embed_tokens.weight.requires_grad_(False)

        return model.to(device)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,         # [B, T] token ids (full vocabulary)
        fused_features: Tensor,    # [B, T, 3*H] concatenated target hidden states
        block_size: int | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            input_ids:      [B, T] full-vocabulary token ids (masked positions contain
                            a designated mask token id, e.g. eos_token_id).
            fused_features: [B, T, 3H] concatenated hidden states from the frozen
                            target model at the 3 extraction layers.
            block_size:     override self.block_size for this call.
            position_ids:   [B, T] (optional; defaults to 0,1,...,T-1).

        Returns:
            logits: [B, T, draft_vocab_size]
        """
        bs = block_size if block_size is not None else self.block_size
        B, T = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        # 1. Project 3 target hidden states → hidden_size
        projected = self.fc(fused_features.to(self.fc.weight.dtype))  # [B, T, H]

        # 2. Norms
        norm_proj = self.hidden_norm(projected)                        # [B, T, H]
        norm_emb  = self.input_layernorm(self.embed_tokens(input_ids)) # [B, T, H]

        # 3. Concatenate → attention input
        x = torch.cat([norm_emb, norm_proj], dim=-1)                   # [B, T, 2H]

        # 4. Block-causal attention mask [1, 1, T, T]
        attn_mask = block_causal_mask(T, bs, device)                   # [T, T]
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)                # [1, 1, T, T]
        attn_mask = attn_mask.to(x.dtype)

        # 5. Self-attention; residual goes around the concat (pre-attn state = projected)
        attn_out = self.self_attn(x, attn_mask, position_ids)          # [B, T, H]

        # 6. Residual: add attn output to the projected target features (not the 2H concat)
        h = projected + attn_out                                       # [B, T, H]

        # 7. MLP with residual
        h = h + self.mlp(self.post_attn_norm(h))                      # [B, T, H]

        # 8. Final norm + LM head
        logits = self.lm_head(self.norm(h))                           # [B, T, draft_vocab_size]
        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def masked_diffusion_loss(
        logits: Tensor,
        targets: Tensor,
        mask_positions: Tensor,
        t2d: Tensor,
        t: float | Tensor,
        use_position_weight: bool = False,
        block_size: int = 1,
        gamma: float = 5.0,
    ) -> Tensor:
        """
        Masked diffusion (BD-LM) cross-entropy loss over the draft vocabulary.

        Only computes loss at positions where:
          (a) the position is masked (mask_positions is True), AND
          (b) the target token is in the draft vocabulary (t2d[target_id] is True).

        Args:
            logits:          [B, T, draft_vocab_size]
            targets:         [B, T]  ground-truth token ids (full vocabulary)
            mask_positions:  [B, T]  bool, True = masked
            t2d:             [full_vocab_size] bool — True if in draft vocab
            t:               scalar noise level (masking rate)
            use_position_weight: DFlash-style exponential decay over block pos
            block_size:      used when use_position_weight=True
            gamma:           decay rate

        Returns: scalar loss
        """
        if not mask_positions.any():
            return logits.sum() * 0.0

        t_val = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=logits.device, dtype=logits.dtype)
        weight = 1.0 / t_val.clamp(min=1e-3)

        B, T, V = logits.shape

        # Map target tokens to draft vocabulary indices.
        # Tokens not in draft vocab → -1 (will be excluded from loss).
        t2d = t2d.to(targets.device)
        in_draft = t2d[targets]                    # [B, T] bool

        # Build draft-vocab target: cumulative sum of t2d gives rank within draft vocab
        # Draft token id for target token x = t2d[:x].sum()
        draft_target_ids = t2d.cumsum(0) - 1      # [full_vocab_size] — 0-indexed draft id
        draft_target_ids[~t2d] = -1               # invalid mapping
        targets_draft = draft_target_ids[targets]  # [B, T]

        # Active positions: masked AND target in draft vocab
        active = mask_positions & in_draft         # [B, T]

        if not active.any():
            return logits.sum() * 0.0

        logits_flat = logits.view(B * T, V)
        targets_flat = targets_draft.view(B * T)
        active_flat = active.view(B * T)

        ce = F.cross_entropy(
            logits_flat[active_flat],
            targets_flat[active_flat],
            reduction="none",
        )  # [n_active]

        if use_position_weight:
            pos_in_block = (torch.arange(T, device=logits.device) % block_size).float()
            pos_weight = torch.exp(-pos_in_block / gamma)  # [T]
            pos_weight = pos_weight.unsqueeze(0).expand(B, T).reshape(B * T)
            ce = ce * pos_weight[active_flat]

        return weight * ce.mean()
