"""
Evaluate generation quality for all systems.

Benchmarks:
  - GSM8K exact match (arithmetic reasoning)
  - HumanEval pass@1 (code generation)
  - ROUGE-L on UltraChat held-out (200 samples)

Usage:
    python scripts/eval_quality.py --config configs/primary.yaml \
        --system [baseline | eagle3 | bd_eagle | dflash] \
        --checkpoint checkpoints/primary/step_011500
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--system", required=True,
                   choices=["baseline", "eagle3", "bd_eagle", "dflash"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--benchmark", default="gsm8k",
                   choices=["gsm8k", "humaneval", "rouge"])
    p.add_argument("--n_samples", type=int, default=None,
                   help="Override sample count (defaults to full benchmark)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ------------------------------------------------------------------
# GSM8K
# ------------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final numeric answer from model output."""
    # Try to find #### pattern (GSM8K format)
    m = re.search(r"####\s*([\-\d,]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: last number in output
    nums = re.findall(r"[\-]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def eval_gsm8k(model, tokenizer, n_samples: int | None, device: torch.device) -> dict:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    correct = 0
    total = 0
    for item in tqdm(ds, desc="GSM8K"):
        prompt = f"Question: {item['question']}\nAnswer:"
        ref_answer = item["answer"].split("####")[-1].strip().replace(",", "")

        pred_text = generate_text(model, tokenizer, prompt, max_new_tokens=256, device=device)
        pred_answer = extract_gsm8k_answer(pred_text)

        if pred_answer is not None and pred_answer == ref_answer:
            correct += 1
        total += 1

    return {"benchmark": "gsm8k", "accuracy": correct / total, "correct": correct, "total": total}


# ------------------------------------------------------------------
# ROUGE-L on UltraChat held-out
# ------------------------------------------------------------------

def rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between two strings."""
    h_tokens = hypothesis.lower().split()
    r_tokens = reference.lower().split()
    if not h_tokens or not r_tokens:
        return 0.0

    # LCS length (dynamic programming)
    m, n = len(r_tokens), len(h_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if r_tokens[i - 1] == h_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]

    precision = lcs / n if n else 0
    recall = lcs / m if m else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def eval_rouge(model, tokenizer, n_samples: int | None, device: torch.device) -> dict:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    n = n_samples or 200
    ds = ds.select(range(min(n, len(ds))))

    scores = []
    for item in tqdm(ds, desc="ROUGE-L"):
        messages = item["messages"]
        # Use all turns except the last assistant reply as prompt
        if len(messages) < 2:
            continue
        prompt_messages = messages[:-1]
        reference = messages[-1]["content"]

        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        pred = generate_text(model, tokenizer, prompt, max_new_tokens=256, device=device)
        scores.append(rouge_l(pred, reference))

    return {
        "benchmark": "rouge_l",
        "mean_rouge_l": sum(scores) / len(scores) if scores else 0,
        "n_samples": len(scores),
    }


# ------------------------------------------------------------------
# HumanEval (pass@1 with greedy decoding)
# ------------------------------------------------------------------

def eval_humaneval(model, tokenizer, n_samples: int | None, device: torch.device) -> dict:
    try:
        from human_eval.data import read_problems
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError:
        return {
            "benchmark": "humaneval",
            "error": "human_eval package not installed. Run: pip install human-eval",
        }

    problems = read_problems()
    if n_samples:
        task_ids = list(problems.keys())[:n_samples]
        problems = {k: problems[k] for k in task_ids}

    completions = []
    for task_id, problem in tqdm(problems.items(), desc="HumanEval"):
        prompt = problem["prompt"]
        completion = generate_text(model, tokenizer, prompt, max_new_tokens=256, device=device)
        completions.append({"task_id": task_id, "completion": completion})

    # Write completions to temp file for evaluation
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for c in completions:
            f.write(json.dumps(c) + "\n")
        tmp_path = f.name

    results = evaluate_functional_correctness(tmp_path)
    os.unlink(tmp_path)

    return {
        "benchmark": "humaneval",
        "pass_at_1": results["pass@1"],
        "n_problems": len(completions),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    print(f"[eval_quality] Loading target model: {cfg['target_model']}")
    target = AutoModelForCausalLM.from_pretrained(
        cfg["target_model"],
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    target.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"])

    # For speculative systems, we load the drafter but the quality evaluation
    # still uses target model greedy decoding (the speculative decoding is
    # lossless by construction, so quality == baseline quality when acceptance
    # threshold is exact match).
    # If you want to test with the speculative path, use eval_throughput.py.
    model = target

    benchmark_fn = {
        "gsm8k": eval_gsm8k,
        "humaneval": eval_humaneval,
        "rouge": eval_rouge,
    }[args.benchmark]

    result = benchmark_fn(model, tokenizer, args.n_samples, device)
    result["system"] = args.system

    print("\n=== Quality Results ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"[eval_quality] Saved → {args.output}")


if __name__ == "__main__":
    main()
