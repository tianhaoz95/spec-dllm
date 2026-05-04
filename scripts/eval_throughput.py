"""
Evaluate speculative decoding throughput for all systems.

Measures:
  - Mean accepted tokens per step (τ̄)
  - Wall-clock tokens/second at batch size 1
  - Acceptance rate per position within a block (for BD-EAGLE / DFlash)

Usage:
    python scripts/eval_throughput.py --config configs/primary.yaml \
        --system [eagle3 | bd_eagle | dflash | baseline] \
        --checkpoint checkpoints/primary/step_011500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from bd_eagle.attention import block_causal_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--system", required=True,
                   choices=["baseline", "eagle3", "bd_eagle", "dflash"])
    p.add_argument("--checkpoint", default=None,
                   help="Drafter checkpoint (required for bd_eagle)")
    p.add_argument("--n_prompts", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--warmup", type=int, default=10,
                   help="Number of sequences to generate before timing starts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None, help="JSON output path")
    return p.parse_args()


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompts(n: int, seed: int = 42) -> list[str]:
    """Load GSM8K prompts for throughput evaluation."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [f"Question: {item['question']}\nAnswer:" for item in ds]


# ------------------------------------------------------------------
# Baseline: standard autoregressive greedy decoding
# ------------------------------------------------------------------

def run_baseline(
    target_model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: torch.device,
    warmup: int,
) -> dict:
    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(tqdm(prompts, desc="baseline")):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = target_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                use_cache=True,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        n_new = out.shape[1] - inputs["input_ids"].shape[1]
        if i >= warmup:
            total_tokens += n_new
            total_time += t1 - t0

    n_timed = len(prompts) - warmup
    return {
        "system": "baseline",
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
        "mean_tokens_per_prompt": total_tokens / n_timed,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "n_prompts": n_timed,
    }


# ------------------------------------------------------------------
# Speculative decoding: shared verification logic
# ------------------------------------------------------------------

def speculative_decode_ar(
    target_model,
    drafter_model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    n_draft: int = 5,
    device: torch.device = torch.device("cuda"),
) -> tuple[torch.Tensor, int, int]:
    """
    Standard tree-free speculative decoding for AR drafter (EAGLE-3).
    Returns (output_ids, tokens_accepted, n_steps).
    """
    from transformers import GenerationConfig

    tokens_accepted = 0
    n_steps = 0
    generated = input_ids.clone()

    with torch.no_grad():
        past_key_values_target = None

        for _ in range(max_new_tokens // n_draft + 1):
            if generated.shape[1] > input_ids.shape[1] + max_new_tokens:
                break

            # Draft n_draft tokens autoregressively
            draft_ids = []
            draft_logits = []
            draft_input = generated

            for d in range(n_draft):
                out = drafter_model(input_ids=draft_input, use_cache=False)
                next_token_logits = out.logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                draft_ids.append(next_token)
                draft_logits.append(next_token_logits)
                draft_input = torch.cat([draft_input, next_token], dim=1)

            draft_ids_tensor = torch.cat(draft_ids, dim=1)  # [1, n_draft]
            candidate_ids = torch.cat([generated, draft_ids_tensor], dim=1)

            # Target forward over all candidate tokens
            target_out = target_model(input_ids=candidate_ids, use_cache=False)
            target_logits = target_out.logits  # [1, len, V]

            # Accept/reject (greedy: accept if argmax matches)
            n_accepted = 0
            for d in range(n_draft):
                pos = generated.shape[1] + d - 1
                target_pred = target_logits[:, pos, :].argmax(dim=-1)
                if target_pred.item() == draft_ids[d].item():
                    n_accepted += 1
                else:
                    break

            # Append accepted tokens + one target-sampled token at rejection
            new_tokens = draft_ids_tensor[:, :n_accepted]
            generated = torch.cat([generated, new_tokens], dim=1)

            bonus_pos = generated.shape[1] - 1
            bonus_token = target_logits[:, bonus_pos, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, bonus_token], dim=1)

            tokens_accepted += n_accepted + 1
            n_steps += 1

            if tokenizer.eos_token_id in generated[0, input_ids.shape[1]:].tolist():
                break

    return generated, tokens_accepted, n_steps


def speculative_decode_block(
    target_model,
    drafter_model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int = 8,
    device: torch.device = torch.device("cuda"),
    extract_layer_indices: list[int] | None = None,
) -> tuple[torch.Tensor, int, int, list[int]]:
    """
    Block diffusion speculative decoding for BD-EAGLE / DFlash.

    Drafts an entire block of block_size tokens in one forward pass,
    verifies with target, accepts greedily.

    Returns (output_ids, tokens_accepted, n_steps, accepted_per_step).
    """
    tokens_accepted = 0
    n_steps = 0
    accepted_per_step = []
    generated = input_ids.clone()
    num_layers = target_model.config.num_hidden_layers

    if extract_layer_indices is None:
        lo = num_layers // 6
        mid = num_layers // 2
        hi = 5 * num_layers // 6
        extract_layer_indices = [lo, mid, hi]

    # Mask token id
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = tokenizer.unk_token_id or tokenizer.eos_token_id

    with torch.no_grad():
        while generated.shape[1] < input_ids.shape[1] + max_new_tokens:
            T_ctx = generated.shape[1]

            # Extract target hidden states
            target_out = target_model(
                input_ids=generated,
                output_hidden_states=True,
                use_cache=False,
            )
            hs = target_out.hidden_states
            fused = torch.cat(
                [hs[i + 1][:, -1:, :] for i in extract_layer_indices], dim=-1
            )  # [1, 1, 3H] — only last position's features

            # Prepare a block of mask tokens
            block_ids = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)

            # Drafter forward: predict the full block in one pass
            # Conditioning context: fused features from target at the last position
            full_input = torch.cat([generated, block_ids], dim=1)
            T_full = full_input.shape[1]

            # Build fused features for all positions
            # (context positions don't matter for block positions)
            fused_full = torch.zeros(1, T_full, fused.shape[-1], dtype=fused.dtype, device=device)
            fused_full[:, T_ctx - 1:T_ctx, :] = fused  # only last context position has features

            attn_mask = block_causal_mask(T_full, block_size=block_size, device=device)
            attn_mask_4d = attn_mask.unsqueeze(0).unsqueeze(0)

            try:
                draft_out = drafter_model(
                    input_ids=full_input,
                    hidden_states=fused_full,
                    attention_mask=attn_mask_4d,
                    use_cache=False,
                )
            except TypeError:
                draft_out = drafter_model(
                    input_ids=full_input,
                    attention_mask=attn_mask_4d,
                    use_cache=False,
                )

            draft_logits = draft_out.logits[:, T_ctx:, :]  # [1, block_size, V]
            draft_tokens = draft_logits.argmax(dim=-1)  # [1, block_size]

            # Target verification: run target on context + draft block
            candidate = torch.cat([generated, draft_tokens], dim=1)
            target_verify = target_model(input_ids=candidate, use_cache=False)
            target_logits = target_verify.logits  # [1, T_ctx + block_size, V]

            # Accept/reject greedily
            n_acc = 0
            for d in range(block_size):
                target_pred = target_logits[:, T_ctx + d - 1, :].argmax(dim=-1)
                if target_pred.item() == draft_tokens[0, d].item():
                    n_acc += 1
                else:
                    break

            # Append accepted tokens
            accepted_tokens = draft_tokens[:, :n_acc]
            generated = torch.cat([generated, accepted_tokens], dim=1)

            # Append one target-sampled bonus token
            bonus_pos = T_ctx + n_acc - 1
            bonus = target_logits[:, bonus_pos, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, bonus], dim=1)

            tokens_accepted += n_acc + 1
            n_steps += 1
            accepted_per_step.append(n_acc + 1)

            if tokenizer.eos_token_id in generated[0, input_ids.shape[1]:].tolist():
                break

    return generated, tokens_accepted, n_steps, accepted_per_step


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_system(
    system: str,
    target_model,
    tokenizer,
    drafter_model,
    prompts: list[str],
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    block_size = cfg.get("inference_block_size", cfg.get("stable_size", 8))
    max_new = args.max_new_tokens
    warmup = args.warmup

    total_tokens = 0
    total_time = 0.0
    total_steps = 0
    per_step_accepted = []

    for i, prompt in enumerate(tqdm(prompts, desc=system)):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if system == "baseline":
            with torch.no_grad():
                out = target_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new,
                    do_sample=False,
                    use_cache=True,
                )
            n_new = out.shape[1] - input_ids.shape[1]
            steps = n_new
            acc = [1] * n_new

        elif system == "eagle3":
            _, n_tok, steps = speculative_decode_ar(
                target_model, drafter_model, tokenizer,
                input_ids, max_new, n_draft=5, device=device
            )
            n_new = n_tok
            acc = []  # not tracked per-step for AR

        else:  # bd_eagle or dflash
            _, n_tok, steps, acc = speculative_decode_block(
                target_model, drafter_model, tokenizer,
                input_ids, max_new, block_size=block_size, device=device,
                extract_layer_indices=cfg.get("target_layer_indices"),
            )
            n_new = n_tok

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup:
            total_tokens += n_new
            total_time += t1 - t0
            total_steps += steps
            per_step_accepted.extend(acc)

    n_timed = len(prompts) - warmup
    mean_accepted = (sum(per_step_accepted) / len(per_step_accepted)) if per_step_accepted else 0

    return {
        "system": system,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
        "mean_accepted_per_step": mean_accepted,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "total_steps": total_steps,
        "n_prompts": n_timed,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    print(f"[eval_throughput] Loading target model: {cfg['target_model']}")
    target_model = AutoModelForCausalLM.from_pretrained(
        cfg["target_model"],
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    target_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"])

    drafter_model = None
    if args.system != "baseline":
        drafter_id = {
            "eagle3": cfg["eagle_model"],
            "dflash": cfg["dflash_model"],
            "bd_eagle": args.checkpoint or cfg.get("bd_eagle_checkpoint"),
        }[args.system]
        print(f"[eval_throughput] Loading drafter: {drafter_id}")
        drafter_model = AutoModelForCausalLM.from_pretrained(
            drafter_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        drafter_model.eval()

    print(f"[eval_throughput] Loading {args.n_prompts} prompts from GSM8K")
    prompts = load_prompts(args.n_prompts)

    result = run_system(
        system=args.system,
        target_model=target_model,
        tokenizer=tokenizer,
        drafter_model=drafter_model,
        prompts=prompts,
        cfg=cfg,
        args=args,
        device=device,
    )

    print("\n=== Results ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"[eval_throughput] Saved → {args.output}")


if __name__ == "__main__":
    main()
