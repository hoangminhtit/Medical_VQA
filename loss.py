import torch
import torch.nn as nn

bce = nn.BCEWithLogitsLoss()
ce  = nn.CrossEntropyLoss(ignore_index=-100)  # -100 = PyTorch convention (dataset replaces pad with -100)


def compute_loss(
        yesno_logits,
    gen_logits,         # Real generation logits from T5 (or None)
        yesno_labels,
        gen_labels,         # T5 tokenized labels  
    is_yesno=None,
        alpha: float = 1.0, # Focus on Y/N classification
    beta:  float = 0.5, # Enable generative loss for T5 training
    is_dummy_logits: bool = False,
):
    """Compute loss for both Yes/No classification and text generation.
    
    Now properly trains T5's text generation capabilities by computing
    cross-entropy loss on generated tokens.
    """
    # Compute BCE only on yes/no samples.
    if is_yesno is None:
        loss_yesno = bce(
            yesno_logits.squeeze(-1),
            yesno_labels.float()
        )
    else:
        is_yesno = is_yesno.to(device=yesno_logits.device, dtype=torch.bool)
        if is_yesno.any():
            loss_yesno = bce(
                yesno_logits[is_yesno].squeeze(-1),
                yesno_labels[is_yesno].float()
            )
        else:
            loss_yesno = torch.tensor(0.0, device=yesno_logits.device)

    # Compute generation loss only if generation logits are provided.
    if gen_logits is None or is_dummy_logits:
        loss_gen = torch.tensor(0.0, device=yesno_logits.device)
    else:
        # Real generation training (open-ended only)
        if is_yesno is not None:
            is_yesno = is_yesno.to(device=gen_logits.device, dtype=torch.bool)
            open_mask = ~is_yesno
        else:
            open_mask = None

        if open_mask is None or open_mask.any():
            if open_mask is not None:
                gen_logits = gen_logits[open_mask]
                gen_labels = gen_labels[open_mask]

            # Reshape for cross-entropy: (B*L, vocab_size) and (B*L,)
            gen_logits_flat = gen_logits.view(-1, gen_logits.size(-1))  # (B*L, vocab_size)
            gen_labels_flat = gen_labels.view(-1)                       # (B*L,)
            loss_gen = ce(gen_logits_flat, gen_labels_flat)
        else:
            loss_gen = torch.tensor(0.0, device=gen_logits.device)

    return alpha * loss_yesno + beta * loss_gen