import argparse
import os
import torch


def get_args():
    """Parse command-line arguments.

    All hyperparameters and paths are configurable via CLI so that the same
    codebase can be run locally, on Kaggle, or on any cluster without editing
    source files — just pass different arguments.

    Example (Kaggle):
        python train.py \\
            --dataset flaviagiammarino/path-vqa \\
            --checkpoint /kaggle/working/checkpoint.pt \\
            --log_dir   /kaggle/working/logs \\
            --log_name  pathvqa_run1 \\
            --batch_size 32 --epochs 20 --device cuda

    Then evaluate on the same log file:
        python test.py \\
            --checkpoint /kaggle/working/checkpoint.pt \\
            --log_dir    /kaggle/working/logs \\
            --log_name   pathvqa_run1          # <-- same name = same log file
    """
    parser = argparse.ArgumentParser(
        description="Medical VQA — BLIP-2 ViT-L + BioBERT + T5-small"
    )

    # ── Dataset ────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default="flaviagiammarino/path-vqa",
        help="HuggingFace dataset identifier"
    )

    # ── Paths ──────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint.pt",
        help="Path to save (train) or load (test) the model checkpoint"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs",
        help="Directory where log files are stored"
    )
    parser.add_argument(
        "--log_name", type=str, default=None,
        help=(
            "Log file base-name (no extension). "
            "Use the SAME name for train.py and test.py to append "
            "train + test results into one shared log file."
        )
    )

    # ── Training hyperparameters ───────────────────────────────────
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1,
        help="Fraction of total training steps used for LR warmup"
    )
    parser.add_argument("--num_workers", type=int,   default=2 if os.name == "nt" else 4)
    parser.add_argument(
        "--grad_accum_steps", type=int, default=1,
        help="Accumulate gradients over N steps to emulate larger batch size"
    )
    parser.add_argument(
        "--use_amp", action="store_true",
        help="Enable automatic mixed precision (CUDA only)"
    )
    parser.add_argument(
        "--pin_memory", action="store_true",
        help="Pin CPU memory in DataLoader (recommended for CUDA)"
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=2,
        help="DataLoader prefetch factor (only if num_workers > 0)"
    )
    parser.add_argument(
        "--persistent_workers", action="store_true",
        help="Keep DataLoader workers alive between epochs"
    )
    parser.add_argument(
        "--allow_tf32", action="store_true",
        help="Enable TF32 matmul on Ampere+ GPUs for speed"
    )

    # ── Hugging Face runtime ──────────────────────────────────────
    parser.add_argument(
        "--hf_cache_dir", type=str, default=None,
        help="Optional cache directory for HF models/datasets"
    )
    parser.add_argument(
        "--hf_offline", action="store_true",
        help="Run in offline mode (requires models/datasets already cached)"
    )
    parser.add_argument(
        "--hf_timeout", type=int, default=120,
        help="HF hub network timeout in seconds"
    )
    parser.add_argument(
        "--show_hf_warnings", action="store_true",
        help="Show Hugging Face hub warnings/logs (default: hidden)"
    )

    # ── Model ──────────────────────────────────────────────────────
    parser.add_argument("--encoder_dim",        type=int, default=768)
    parser.add_argument("--max_answer_len",      type=int, default=24)
    parser.add_argument(
        "--t5_model", type=str, default="t5-small",
        help="T5 model name or path (e.g., google/flan-t5-small)"
    )
    parser.add_argument("--image_unfreeze_top",  type=int, default=12,
        help="Number of top ViT-L layers to unfreeze")
    parser.add_argument("--text_unfreeze_top",   type=int, default=12,
        help="Number of top BioBERT layers to unfreeze")
    parser.add_argument(
        "--gen_num_beams", type=int, default=4,
        help="Beam size for T5 generation"
    )
    parser.add_argument(
        "--gen_repetition_penalty", type=float, default=1.2,
        help="Repetition penalty for T5 generation"
    )
    parser.add_argument(
        "--gen_no_repeat_ngram_size", type=int, default=2,
        help="No-repeat ngram size for T5 generation"
    )

    # ── Loss weights ───────────────────────────────────────────────
    parser.add_argument(
        "--loss_alpha", type=float, default=1.0,
        help="Weight for Yes/No BCE loss"
    )
    parser.add_argument(
        "--loss_beta", type=float, default=1.0,
        help="Weight for generative CE loss (now enabled for T5 training)"
    )

    # ── Early stopping ─────────────────────────────────────────────
    parser.add_argument(
        "--early_stopping", type=int, default=5,
        help="Stop training if val Y/N acc has not improved for this many epochs (0 = disabled)"
    )

    # ── Device ─────────────────────────────────────────────────────
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


class DatasetConfig:
    """Hardcoded HuggingFace dataset identifiers used as defaults."""
    PATH_VQA = "flaviagiammarino/path-vqa"
    VQA_MED  = "flaviagiammarino/vqa-rad"   # VQA-Med 2019 proxy