import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import T5Tokenizer

from config import get_args
from model import MedicalVQAModel
from dataset import MedicalVQADataset
from utils import load_checkpoint
from logger import setup_logger
from metrics import evaluate_medical_vqa
from hf_runtime import configure_hf_runtime


def evaluate_test(args):
    """Load a trained checkpoint and evaluate on the test split.

    Metrics reported:
        - Yes/No accuracy
        - Open-ended exact-match accuracy
        - BLEU-1, BLEU-2, BLEU-3, BLEU-4
        - Composite BLEU with brevity penalty

    Pass --log_name matching the training run to append results into the
    same log file produced by train.py.
    """
    logger = setup_logger(args.log_dir, args.log_name)
    configure_hf_runtime(args)
    device = args.device

    logger.info("=" * 60)
    logger.info("Test evaluation started")
    logger.info(f"  Dataset    : {args.dataset}")
    logger.info(f"  Checkpoint : {args.checkpoint}")
    logger.info(f"  T5 model   : {args.t5_model}")
    logger.info(f"  Device     : {device}")
    logger.info("=" * 60)

    # ── Model ──────────────────────────────────────────────────────
    model = MedicalVQAModel(
        dim=args.encoder_dim,
        max_answer_len=args.max_answer_len,
        image_unfreeze_top=args.image_unfreeze_top,
        text_unfreeze_top=args.text_unfreeze_top,
        t5_model=args.t5_model,
        gen_num_beams=args.gen_num_beams,
        gen_repetition_penalty=args.gen_repetition_penalty,
        gen_no_repeat_ngram_size=args.gen_no_repeat_ngram_size,
    ).to(device)

    load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()
    logger.info("Checkpoint loaded successfully.")

    # ── Tokenizer ──────────────────────────────────────────────────
    # T5 tokenizer matches the decoder vocabulary (answers encoded by dataset)
    logger.info("Loading tokenizer...")
    t5_tokenizer = T5Tokenizer.from_pretrained(args.t5_model)

    # ── Dataset ────────────────────────────────────────────────────
    logger.info("Loading dataset from Hugging Face...")
    ds = load_dataset(args.dataset)
    test_split = ds.get("test") or ds.get("validation") or ds.get("val")
    if test_split is None:
        raise ValueError(
            f"Dataset '{args.dataset}' has no 'test', 'validation', or 'val' split. "
            f"Available splits: {list(ds.keys())}"
        )
    test_ds    = MedicalVQADataset(test_split, max_answer_len=args.max_answer_len)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    logger.info(f"Test samples : {len(test_ds)}")

    # ── Comprehensive Evaluation ───────────────────────────────────
    logger.info("Running evaluation...")

    results = evaluate_medical_vqa(
        model=model,
        dataloader=test_loader,
        device=device,
        t5_tokenizer=t5_tokenizer,
        verbose=True
    )

    # ── Log to file ────────────────────────────────────────────────
    sep = "=" * 60
    logger.info(sep)
    logger.info("Test Set Results")
    logger.info(sep)

    if results["yesno_accuracy"] is not None:
        logger.info(f"  Yes/No Accuracy      : {results['yesno_accuracy']:.4f}")
    else:
        logger.info(f"  Yes/No Accuracy      : N/A")

    logger.info(f"  Open Exact Match     : {results['open_exact_match']:.4f}")
    logger.info("-" * 60)
    logger.info(f"  BLEU-1               : {results['bleu1']:.4f}")
    logger.info(f"  BLEU-2               : {results['bleu2']:.4f}")
    logger.info(f"  BLEU-3               : {results['bleu3']:.4f}")
    logger.info(f"  BLEU-4               : {results['bleu4']:.4f}")
    logger.info(f"  BLEU Composite       : {results['bleu_composite']:.4f}")
    logger.info(f"  Brevity Penalty      : {results['brevity_penalty']:.4f}")
    logger.info(sep)

    return {
        "yesno_acc"     : results["yesno_accuracy"],
        "open_exact"    : results["open_exact_match"],
        "bleu1"         : results["bleu1"],
        "bleu2"         : results["bleu2"],
        "bleu3"         : results["bleu3"],
        "bleu4"         : results["bleu4"],
        "bleu_composite": results["bleu_composite"],
        "brevity_penalty": results["brevity_penalty"],
    }


if __name__ == "__main__":
    args = get_args()
    evaluate_test(args)
