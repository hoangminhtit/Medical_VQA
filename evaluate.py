import torch
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def evaluate(model, dataloader, device):
    """Evaluate on a dataloader.

    Returns a dict with:
        yesno_acc  – accuracy on yes/no questions
        open_bleu  – mean BLEU-1 on open-ended questions
        open_exact – exact-match accuracy on open-ended questions
    """
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    smoother = SmoothingFunction().method1

    yesno_correct = 0
    yesno_total   = 0
    open_bleu     = 0.0
    open_exact    = 0
    open_total    = 0

    with torch.no_grad():
        for batch in dataloader:
            images    = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            mask      = batch["attention_mask"].to(device)
            yn_labels = batch["yesno"].to(device)
            is_yn     = batch["is_yesno"].bool()
            gen_lbl   = batch["answer"]

            yesno_logits, _ = model(images, input_ids, mask, generate_text=False)  # Classification mode
            
            # Generate text for open-ended questions
            _, generated_ids = model(images, input_ids, mask, generate_text=True)   # Generation mode
            
            # Use T5 tokenizer for generated text
            from transformers import T5Tokenizer
            t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

            # Yes / No accuracy
            if is_yn.any():
                preds = (yesno_logits[is_yn] > 0).long().squeeze(-1)
                gt    = yn_labels[is_yn]
                yesno_correct += (preds == gt).sum().item()
                yesno_total   += gt.size(0)

            # Open-ended: BLEU-1 + exact match
            open_mask = ~is_yn
            if open_mask.any():
                # Use generated text for open-ended questions
                pred_texts = t5_tokenizer.batch_decode(
                    generated_ids[open_mask].cpu(), skip_special_tokens=True
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
                        open_exact += int(pred.strip().lower() == gt.strip().lower())
                        open_total += 1

    return {
        "yesno_acc" : yesno_correct / yesno_total if yesno_total > 0 else 0.0,
        "open_bleu" : open_bleu    / open_total   if open_total  > 0 else 0.0,
        "open_exact": open_exact   / open_total   if open_total  > 0 else 0.0,
    }