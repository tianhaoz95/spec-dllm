"""
Download all required model checkpoints from HuggingFace.

Models:
  1. Qwen/Qwen3-8B-Instruct          (target model, ~16 GB BF16)
  2. AngelSlim/Qwen3-8B_eagle3       (EAGLE-3 drafter, ~400 MB)
  3. z-lab/Qwen3-8B-DFlash-b16       (DFlash baseline drafter, ~400 MB)

Usage:
    python scripts/download_models.py
"""

from huggingface_hub import snapshot_download
import sys

MODELS = [
    # Base model (not instruct) — matches AngelSlim/Qwen3-8B_eagle3 training setup
    ("Qwen/Qwen3-8B", "target model"),
    ("AngelSlim/Qwen3-8B_eagle3", "EAGLE-3 drafter"),
    ("z-lab/Qwen3-8B-DFlash-b16", "DFlash drafter"),
]

for repo_id, desc in MODELS:
    print(f"\n[download] {desc}: {repo_id}", flush=True)
    try:
        path = snapshot_download(repo_id=repo_id, resume_download=True)
        print(f"[download] OK → {path}", flush=True)
    except Exception as e:
        print(f"[download] FAILED: {e}", file=sys.stderr, flush=True)

print("\n[download] All done.")
