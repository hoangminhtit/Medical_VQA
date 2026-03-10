import torch.nn as nn

bce = nn.BCEWithLogitsLoss()
ce  = nn.CrossEntropyLoss(ignore_index=0)   # 0 = padding token id


def compute_loss(
        yesno_logits,
        gen_logits,
        yesno_labels,
        gen_labels,
        alpha: float = 0.5,
        beta:  float = 0.5
):
    loss_yesno = bce(
        yesno_logits.squeeze(-1),
        yesno_labels.float()
    )

    # gen_logits : (B, S, vocab_size) → (B*S, vocab_size)
    # gen_labels : (B, S)             → (B*S,)
    B, S, V = gen_logits.shape
    loss_gen = ce(
        gen_logits.reshape(B * S, V),
        gen_labels.reshape(B * S).long()
    )

    return alpha * loss_yesno + beta * loss_gen