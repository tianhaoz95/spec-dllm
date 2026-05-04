# BD-EAGLE: Block Diffusion Adaptation of EAGLE-3 Drafters

This project implements **BD-EAGLE**, a speculative decoding drafter that adapts the autoregressive EAGLE-3 architecture into a block diffusion model. It uses a Warmup-Stable-Decay (WSD) fine-tuning curriculum to convert a pre-trained EAGLE-3 checkpoint into a Block Diffusion LM (BD-LM) without architectural changes, only modifying the attention mask to be block-causal.

## Project Overview

- **Core Hypothesis:** Initializing from a pre-trained autoregressive EAGLE-3 checkpoint allows for rapid adaptation (≤30K samples) to a block diffusion objective, achieving high acceptance rates with significantly less data than training from scratch.
- **Target Model:** Qwen3-8B-Instruct (frozen).
- **Drafter:** A single-layer Transformer adapted from `AngelSlim/Qwen3-8B_eagle3`.
- **Infrastructure:** Optimized for NVIDIA Grace Blackwell (GB10) hardware using unified memory.

## Directory Structure

- `bd_eagle/`: Core implementation of the BD-EAGLE model and training logic.
    - `model.py`: `BDEagleDrafter` class; implements feature fusion and block-causal forward pass.
    - `attention.py`: Implementation of the block-causal attention mask.
    - `dataset.py`: Masking logic for the Block Diffusion LM objective.
    - `wsd_scheduler.py`: WSD curriculum logic (ramping block size and LR).
- `scripts/`: Operational scripts for the training and evaluation pipeline.
    - `extract_features.py`: Caches hidden states from the frozen target LLM (highly recommended for training speed).
    - `train_bd_eagle.py`: Main WSD training loop.
    - `eval_throughput.py`: Measures inference speed (tokens/s) and acceptance rate ($\bar{\tau}$).
    - `eval_quality.py`: Evaluates model quality on GSM8K, HumanEval, and MT-Bench.
- `configs/`: YAML configurations for experiments (Primary, Cold Start, Anchor Masking).
- `eagle_lib/`: Reference implementation of the original EAGLE repository.

## Building and Running

### Environment
The project uses a Python 3.12 virtual environment (`.venv/`) with PyTorch, Transformers, and Accelerate.

### GPU Job Management (Task Spooler)
To ensure multiple jobs can share GPU resources without collision, all GPU-intensive tasks (feature extraction, training, evaluation) **MUST** be launched using `task-spooler` by prefixing commands with `tsp`.

- **Queue a Job:** `tsp python scripts/...`
- **View Queue:** `tsp`
- **Monitor Live Output:** `tsp -t [job_id]` (defaults to the last job)
- **View Full Output:** `tsp -c [job_id]`
- **Remove/Kill Job:** `tsp -r [job_id]` (remove) or `tsp -k [job_id]` (kill)
- **Wait for Job:** `tsp -w [job_id]`
- **Set Max Parallel Jobs:** `tsp -S [N]` (e.g., `tsp -S 1` for sequential execution)

### Pipeline Steps
1.  **Feature Extraction:**
    ```bash
    tsp python scripts/extract_features.py --model Qwen/Qwen3-8B-Instruct --output_dir data/feature_cache
    ```
2.  **Training (WSD):**
    ```bash
    tsp python scripts/train_bd_eagle.py --config configs/primary.yaml
    ```
3.  **Evaluation:**
    ```bash
    # Throughput and Acceptance Rate
    tsp python scripts/eval_throughput.py --model_path checkpoints/bd_eagle/step_011500
    
    # Task Quality
    tsp python scripts/eval_quality.py --model_path checkpoints/bd_eagle/step_011500 --task gsm8k
    ```

## Development Conventions

- **Surgical Updates:** When modifying `BDEagleDrafter`, ensure compatibility with the original EAGLE-3 checkpoint loading logic in `model.py`.
- **Lossless Verification:** Speculative decoding must maintain the output distribution of the target model. Always verify that `eval_quality.py` shows zero or near-zero degradation compared to greedy decoding.
- **Efficiency:** The training loop relies on cached features to bypass the expensive frozen target LLM forward pass. When adding new datasets, update `extract_features.py` first.
- **Checkpointing:** Checkpoints save only trainable drafter weights to save space. Use `BDEagleDrafter.from_pretrained` to reload them alongside the frozen embeddings.
