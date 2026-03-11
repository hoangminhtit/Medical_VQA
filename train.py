import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from config import get_args
from dataset import MedicalVQADataset
from model import MedicalVQAModel
from loss import compute_loss
from utils import save_checkpoint
from evaluate import evaluate
from logger import setup_logger


def train(args):
    logger = setup_logger(args.log_dir, args.log_name)
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
    ds      = load_dataset(args.dataset)
    val_key = "validation" if "validation" in ds else "val"

    train_dataset = MedicalVQADataset(ds["train"])
    val_dataset   = MedicalVQADataset(ds[val_key])

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

    # ── Model ────────────────────────────────────────────────────────
    model = MedicalVQAModel(
        encoder_dim=args.encoder_dim,
        vocab_size=args.vocab_size,
        max_answer_len=args.max_answer_len
    ).to(device)

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
    best_yn_acc = 0.0

    for epoch in range(args.epochs):

        model.train()
        total_loss = 0.0

        for batch in train_loader:
            images    = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            mask      = batch["attention_mask"].to(device)
            yn_lbl    = batch["yesno"].to(device)
            gen_lbl   = batch["answer"].to(device)

            yesno_logits, gen_logits = model(
                images, input_ids, mask, labels=gen_lbl, generate_text=False
            )

            loss = compute_loss(
                yesno_logits, gen_logits, yn_lbl, gen_lbl,
                alpha=args.loss_alpha, beta=args.loss_beta
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        metrics  = evaluate(model, val_loader, device)

        # Enhanced logging with comprehensive BLEU metrics
        logger.info(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Loss {avg_loss:.4f} | "
            f"Val Y/N {metrics['yesno_acc']:.4f} | "
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

        # Save best checkpoint based on yes/no accuracy
        if metrics["yesno_acc"] >= best_yn_acc:
            best_yn_acc = metrics["yesno_acc"]
            save_checkpoint(model, args.checkpoint)
            logger.info(f"  → Checkpoint saved (best Y/N Acc {best_yn_acc:.4f})")

    logger.info("Training complete.")


if __name__ == "__main__":
    args = get_args()
    train(args)