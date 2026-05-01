import torch
import torch.nn as nn 

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, query, key, value):

        out, _ = self.attn(query, key, value)

        return out
    
class DualGatingModule(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.text_to_image_attn = CrossAttention(dim)
        self.image_to_text_attn = CrossAttention(dim)

        self.gate1 = nn.Linear(dim, dim)
        self.gate2 = nn.Linear(dim, dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, V, Q):

        # CLS token for image
        v_cls = V[:, 0]

        # CLS token for question (BERT-style, avoids noise from padding tokens)
        q = Q[:, 0]

        # Gate 1 : Text -> Image
        g1 = self.sigmoid(self.gate1(q)).unsqueeze(1)

        cross_V = self.text_to_image_attn(V, Q, Q)

        V_out = V + g1 * cross_V

        # Gate 2 : Image -> Text
        g2 = self.sigmoid(self.gate2(v_cls)).unsqueeze(1)

        cross_Q = self.image_to_text_attn(Q, V, V)

        Q_out = Q + g2 * cross_Q

        return V_out, Q_out