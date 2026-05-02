import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import load_dataset
from config import get_args
from dataset import MedicalVQADataset
from model import MedicalVQAModel
from loss import compute_loss
from utils import save_checkpoint
from evaluate import evaluate
from logger import setup_logger
from hf_runtime import configure_hf_runtime


def train(args):
    logger = setup_logger(args.log_dir, args.log_name)
    configure_hf_runtime(args)
    device = args.device

    logger.info("=" * 60)
    logger.info("Training started")
    logger.info(f"  Dataset     : {args.dataset}")
    logger.info(f"  Device      : {device}")
    logger.info(f"  Epochs      : {args.epochs}")
    logger.info(f"  Batch size  : {args.batch_size}")
    logger.info(f"  LR          : {args.lr}")
    logger.info(f"  Loss alpha  : {args.loss_alpha}  beta: {args.loss_beta}")
    logger.info(f"  Checkpoint  : {args.checkpoint}")
    logger.info("=" * 60)

    # ── Data ────────────────────────────────────────────────────────
    logger.info("Loading dataset from Hugging Face...")
    ds = load_dataset(args.dataset)
    logger.info("Dataset loaded.")

    if "train" not in ds:
        raise ValueError(
            f"Dataset '{args.dataset}' does not contain a 'train' split. Available splits: {list(ds.keys())}"
        )

    train_split = ds["train"]
    val_source = None

    if "validation" in ds:
        val_split = ds["validation"]
        val_source = "validation"
    elif "val" in ds:
        val_split = ds["val"]
        val_source = "val"
    elif "test" in ds:
        val_split = ds["test"]
        val_source = "test"
    else:
        split_ds = train_split.train_test_split(test_size=0.1, seed=42)
        train_split = split_ds["train"]
        val_split = split_ds["test"]
        val_source = "train[10% split]"

    train_dataset = MedicalVQADataset(train_split, max_answer_len=args.max_answer_len)
    val_dataset = MedicalVQADataset(val_split, max_answer_len=args.max_answer_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    logger.info(
        f"Train samples : {len(train_dataset)} | "
        f"Val samples : {len(val_dataset)}"
    )
    logger.info(f"Validation source split : {val_source}")

    # ── Model ────────────────────────────────────────────────────────
    logger.info("Loading model components (BLIP-2 ViT-L + BioBERT + T5-small)...")
    model = MedicalVQAModel(
        dim=args.encoder_dim,
        max_answer_len=args.max_answer_len
    ).to(device)
    logger.info("Model loaded.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params : {trainable:,} / {total:,}")

    # Only optimize trainable parameters (T5 decoder + heads, encoder frozen)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    # ── Training loop ────────────────────────────────────────────────
    best_composite = 0.0
    no_improve_count = 0
    epoch_losses = []

    for epoch in range(args.epochs):

        model.train()
        total_loss = 0.0

        for batch in train_loader:
            pixel_values = batch["image"].to(device)
            input_ids    = batch["input_ids"].to(device)
            mask         = batch["attention_mask"].to(device)
            yn_lbl       = batch["yesno"].to(device)
            is_yn        = batch["is_yesno"].to(device)
            gen_lbl      = batch["answer"].to(device)

            yesno_logits, gen_logits = model(
                pixel_values, input_ids, mask, labels=gen_lbl, generate_text=False
            )

            loss = compute_loss(
                yesno_logits, gen_logits, yn_lbl, gen_lbl,
                is_yesno=is_yn,
                alpha=args.loss_alpha, beta=args.loss_beta
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        metrics  = evaluate(model, val_loader, device)

        # Enhanced logging with comprehensive BLEU metrics
        yn_acc = metrics["yesno_acc"]
        yn_text = f"{yn_acc:.4f}" if yn_acc is not None else "N/A"
        logger.info(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Loss {avg_loss:.4f} | "
            f"Val Y/N {yn_text} | "
            f"Val Exact {metrics['open_exact']:.4f}"
        )
        logger.info(
            f"  BLEU-1: {metrics['bleu1']:.4f} | BLEU-2: {metrics['bleu2']:.4f} | "
            f"BLEU-3: {metrics['bleu3']:.4f} | BLEU-4: {metrics['bleu4']:.4f}"
        )
        logger.info(
            f"  BLEU Composite: {metrics['bleu_composite']:.4f} | "
            f"Brevity Penalty: {metrics['brevity_penalty']:.4f}"
        )

        # ── Composite metric: 50% Y/N acc + 50% BLEU composite ────────
        # Ensures both classification AND generation quality are
        # considered when selecting the best checkpoint.
        yn_acc_val = yn_acc if yn_acc is not None else 0.0
        bleu_comp  = metrics["bleu_composite"]
        composite  = 0.5 * yn_acc_val + 0.5 * bleu_comp

        if composite > best_composite:
            best_composite = composite
            no_improve_count = 0
            save_checkpoint(model, args.checkpoint)
            logger.info(
                f"  → Checkpoint saved "
                f"(composite {composite:.4f} | Y/N {yn_acc_val:.4f} | BLEU {bleu_comp:.4f})"
            )
        else:
            no_improve_count += 1
            logger.info(
                f"  → No improvement for {no_improve_count}/{args.early_stopping} epoch(s) "
                f"(composite {composite:.4f})"
            )
            if args.early_stopping > 0 and no_improve_count >= args.early_stopping:
                logger.info(
                    f"Early stopping triggered after {no_improve_count} epoch(s) without improvement."
                )
                break

    logger.info("Training complete.")

    if epoch_losses:
        log_name = args.log_name or "loss_curve"
        plot_path = os.path.join(args.log_dir, f"{log_name}_loss.png")
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Loss curve saved to: {plot_path}")


if __name__ == "__main__":
    args = get_args()
    train(args)