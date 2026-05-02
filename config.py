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
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1,
        help="Fraction of total training steps used for LR warmup"
    )
    parser.add_argument("--num_workers", type=int,   default=0 if os.name == "nt" else 4)

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
    parser.add_argument("--image_unfreeze_top",  type=int, default=6,
        help="Number of top ViT-L layers to unfreeze")
    parser.add_argument("--text_unfreeze_top",   type=int, default=6,
        help="Number of top BioBERT layers to unfreeze")

    # ── Loss weights ───────────────────────────────────────────────
    parser.add_argument(
        "--loss_alpha", type=float, default=1.0,
        help="Weight for Yes/No BCE loss"
    )
    parser.add_argument(
        "--loss_beta", type=float, default=1.5,
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