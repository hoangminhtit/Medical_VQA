import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """Single-direction cross-attention with pre-LayerNorm and dropout."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, Nq, dim)
            key:   (B, Nk, dim)
            value: (B, Nk, dim)
        Returns:
            out:   (B, Nq, dim)  — cross-attended, post-dropout
        """
        q  = self.norm_q(query)
        k  = self.norm_kv(key)
        v  = self.norm_kv(value)
        out, _ = self.attn(q, k, v)
        return self.dropout(out)


class DualGatingModule(nn.Module):
    """Question-Aware Dual Gating with Pre-LayerNorm gates and CrossAttn dropout.

    Improvements over the original:
      - LayerNorm applied to gate inputs (pre-norm for training stability)
      - Dropout(0.1) applied after each CrossAttention output
      - Residual connections preserved (V' = V + g1 * cross_V)
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Pre-norm for gate signal computation
        self.gate1_norm = nn.LayerNorm(dim)   # normalise Q CLS before gate1
        self.gate2_norm = nn.LayerNorm(dim)   # normalise V CLS before gate2

        # Cross-attention modules (each includes pre-norm + dropout internally)
        self.text_to_image_attn = CrossAttention(dim, num_heads, dropout)  # Gate 1: Text→Image
        self.image_to_text_attn = CrossAttention(dim, num_heads, dropout)  # Gate 2: Image→Text

        # Gating linear layers
        self.gate1 = nn.Linear(dim, dim)
        self.gate2 = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, V: torch.Tensor, Q: torch.Tensor):
        """
        Args:
            V: (B, Nv, dim)  — image patch tokens
            Q: (B, Nq, dim)  — question tokens
        Returns:
            V_out: (B, Nv, dim)
            Q_out: (B, Nq, dim)
        """
        # ── Gate 1 : Question guides Image attention ─────────────────────
        # Use CLS token of Q (position 0 = [CLS] in BERT-style)
        q_cls  = self.gate1_norm(Q[:, 0])                 # (B, dim)
        g1     = self.sigmoid(self.gate1(q_cls)).unsqueeze(1)  # (B, 1, dim)
        cross_V = self.text_to_image_attn(V, Q, Q)        # (B, Nv, dim)
        V_out  = V + g1 * cross_V

        # ── Gate 2 : Image guides Question attention ─────────────────────
        # Use CLS token of V (position 0 = CLS patch)
        v_cls  = self.gate2_norm(V[:, 0])                 # (B, dim)
        g2     = self.sigmoid(self.gate2(v_cls)).unsqueeze(1)  # (B, 1, dim)
        cross_Q = self.image_to_text_attn(Q, V, V)        # (B, Nq, dim)
        Q_out  = Q + g2 * cross_Q

        return V_out, Q_out