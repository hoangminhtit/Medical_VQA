import torch
from transformers import T5Tokenizer
from metrics import evaluate_medical_vqa


def evaluate(model, dataloader, device):
    """Evaluate on a dataloader with comprehensive metrics.

    Returns a dict with:
        yesno_accuracy     – accuracy on yes/no questions
        open_exact_match   – exact-match accuracy on open-ended questions  
        bleu1, bleu2, bleu3, bleu4 – individual BLEU scores
        bleu_composite     – composite BLEU score with brevity penalty
        brevity_penalty    – brevity penalty factor
    """
    model.eval()

    # Use T5 tokenizer for answer decoding (matches dataset)
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    # Use comprehensive evaluation from metrics module
    results = evaluate_medical_vqa(
        model=model,
        dataloader=dataloader, 
        device=device,
        t5_tokenizer=t5_tokenizer,
        verbose=False  # Don't print during training
    )
    
    # Return in format expected by training loop
    return {
        "yesno_acc": results["yesno_accuracy"] or 0.0,
        "open_bleu": results["bleu1"],  # For compatibility with training loop
        "open_exact": results["open_exact_match"],
        # Additional comprehensive metrics
        "bleu1": results["bleu1"], 
        "bleu2": results["bleu2"],
        "bleu3": results["bleu3"], 
        "bleu4": results["bleu4"],
        "bleu_composite": results["bleu_composite"],
        "brevity_penalty": results["brevity_penalty"]
    }