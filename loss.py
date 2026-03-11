import torch
import torch.nn as nn

bce = nn.BCEWithLogitsLoss()
ce  = nn.CrossEntropyLoss(ignore_index=0)   # Keep for compatibility


def compute_loss(
        yesno_logits,
        gen_logits,         # Dummy logits from T5 - not used
        yesno_labels,
        gen_labels,         # Dummy labels - not used  
        alpha: float = 1.0, # Focus on Y/N classification
        beta:  float = 0.0  # Disable generative loss (T5 handles internally)
):
    """Compute loss with focus on Yes/No classification.
    
    T5 handles text generation training internally through its language modeling
    head, so we only train the Yes/No classification head explicitly.
    """
    loss_yesno = bce(
        yesno_logits.squeeze(-1),
        yesno_labels.float()
    )

    # Note: gen_logits are dummy tensors, so gen loss is always 0
    # T5 will learn text generation through its internal training
    loss_gen = torch.tensor(0.0, device=yesno_logits.device)

    return alpha * loss_yesno + beta * loss_gen