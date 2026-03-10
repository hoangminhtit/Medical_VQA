import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from config import get_args
from model import MedicalVQAModel
from dataset import MedicalVQADataset
from utils import load_checkpoint
from logger import setup_logger


def evaluate_test(args):
    """Load a trained checkpoint and evaluate on the test split.

    Metrics reported:
        - Yes/No accuracy
        - Open-ended BLEU-1
        - Open-ended exact-match accuracy

    Pass --log_name matching the training run to append results into the
    same log file produced by train.py.
    """
    logger = setup_logger(args.log_dir, args.log_name)
    device = args.device

    logger.info("=" * 60)
    logger.info("Test evaluation started")
    logger.info(f"  Dataset    : {args.dataset}")
    logger.info(f"  Checkpoint : {args.checkpoint}")
    logger.info(f"  Device     : {device}")
    logger.info("=" * 60)

    # ── Model ──────────────────────────────────────────────────────
    model = MedicalVQAModel(
        encoder_dim=args.encoder_dim,
        vocab_size=args.vocab_size,
        max_answer_len=args.max_answer_len
    ).to(device)

    load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()
    logger.info("Checkpoint loaded successfully.")

    # ── Tokenizer (decodes generated answer token IDs → text) ──────
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    # ── Dataset ────────────────────────────────────────────────────
    ds         = load_dataset(args.dataset)
    test_split = ds.get("test", ds.get("validation"))
    test_ds    = MedicalVQADataset(test_split)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    logger.info(f"Test samples : {len(test_ds)}")

    # ── Metrics ────────────────────────────────────────────────────
    smoother = SmoothingFunction().method1

    yesno_correct = 0
    yesno_total   = 0
    open_bleu     = 0.0
    open_exact    = 0
    open_total    = 0

    with torch.no_grad():
        for batch in test_loader:
            images    = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            mask      = batch["attention_mask"].to(device)
            yn_labels = batch["yesno"].to(device)
            is_yn     = batch["is_yesno"].bool()
            gen_lbl   = batch["answer"]     # stays on CPU for decoding

            yesno_logits, gen_logits = model(images, input_ids, mask)

            # Yes / No
            if is_yn.any():
                preds = (yesno_logits[is_yn] > 0).long().squeeze(-1)
                gt    = yn_labels[is_yn]
                yesno_correct += (preds == gt).sum().item()
                yesno_total   += gt.size(0)

            # Open-ended
            open_mask = ~is_yn
            if open_mask.any():
                pred_ids   = gen_logits[open_mask].argmax(dim=-1)   # (N, 16)
                pred_texts = tokenizer.batch_decode(
                    pred_ids.cpu(), skip_special_tokens=True
                )
                gt_texts = tokenizer.batch_decode(
                    gen_lbl[open_mask].cpu(), skip_special_tokens=True
                )
                for pred, gt in zip(pred_texts, gt_texts):
                    pred_tok = pred.lower().split()
                    gt_tok   = gt.lower().split()
                    if gt_tok:
                        open_bleu  += sentence_bleu(
                            [gt_tok], pred_tok, smoothing_function=smoother
                        )
                        open_exact += int(
                            pred.strip().lower() == gt.strip().lower()
                        )
                        open_total += 1

    # ── Results ────────────────────────────────────────────────────
    yn_acc    = yesno_correct / yesno_total if yesno_total > 0 else None
    avg_bleu  = open_bleu    / open_total   if open_total  > 0 else None
    avg_exact = open_exact   / open_total   if open_total  > 0 else None

    sep = "=" * 60
    logger.info(sep)
    logger.info("Test Set Results")
    logger.info(sep)

    if yn_acc is not None:
        logger.info(
            f"  Yes/No  Accuracy : {yn_acc:.4f}  "
            f"({yesno_correct}/{yesno_total})"
        )
    else:
        logger.info("  Yes/No  Accuracy : N/A")

    if avg_bleu is not None:
        logger.info(f"  Open    BLEU-1   : {avg_bleu:.4f}")
        logger.info(
            f"  Open    Exact    : {avg_exact:.4f}  "
            f"({open_exact}/{open_total})"
        )
    else:
        logger.info("  Open-ended       : N/A")

    logger.info(sep)

    return {
        "yesno_acc" : yn_acc,
        "open_bleu" : avg_bleu,
        "open_exact": avg_exact,
    }


if __name__ == "__main__":
    args = get_args()
    evaluate_test(args)
