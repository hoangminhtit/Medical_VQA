import torch
import torch.nn as nn

bce = nn.BCEWithLogitsLoss()
ce  = nn.CrossEntropyLoss(ignore_index=0)   # Ignore padding tokens (T5 pad_token_id = 0)


def compute_loss(
        yesno_logits,
        gen_logits,         # Real generation logits from T5
        yesno_labels,
        gen_labels,         # T5 tokenized labels  
        alpha: float = 1.0, # Focus on Y/N classification
        beta:  float = 0.5  # Enable generative loss for T5 training
):
    """Compute loss for both Yes/No classification and text generation.
    
    Now properly trains T5's text generation capabilities by computing
    cross-entropy loss on generated tokens.
    """
    loss_yesno = bce(
        yesno_logits.squeeze(-1),
        yesno_labels.float()
    )

    # Compute generation loss only if gen_logits are real (not dummy)
    # Check if gen_logits are dummy by seeing if they're all zeros
    if gen_logits.sum().item() == 0.0:
        # Dummy logits case (compatibility mode)
        loss_gen = torch.tensor(0.0, device=yesno_logits.device)
    else:
        # Real generation training
        # Reshape for cross-entropy: (B*L, vocab_size) and (B*L,)
        gen_logits_flat = gen_logits.view(-1, gen_logits.size(-1))  # (B*L, vocab_size)
        gen_labels_flat = gen_labels.view(-1)                       # (B*L,)
        
        loss_gen = ce(gen_logits_flat, gen_labels_flat)

    return alpha * loss_yesno + beta * loss_gen