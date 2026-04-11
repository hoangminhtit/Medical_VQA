import argparse
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
        description="Medical VQA — BioMedCLIP + T5"
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
    parser.add_argument("--num_workers", type=int,   default=4)

    # ── Model ──────────────────────────────────────────────────────
    parser.add_argument("--encoder_dim",    type=int, default=768)
    parser.add_argument("--vocab_size",     type=int, default=32128)
    parser.add_argument("--max_answer_len", type=int, default=16)

    # ── Loss weights ───────────────────────────────────────────────
    parser.add_argument(
        "--loss_alpha", type=float, default=1.0,
        help="Weight for Yes/No BCE loss"
    )
    parser.add_argument(
        "--loss_beta", type=float, default=0.5,
        help="Weight for generative CE loss (now enabled for T5 training)"
    )

    # ── Early stopping ─────────────────────────────────────────────
    parser.add_argument(
        "--early_stopping", type=int, default=3,
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