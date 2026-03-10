import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

from feature_extraction import BioMedCLIPEncoder
from dual_gating_attention import DualGatingModule


class T5Decoder(nn.Module):
    """T5-base encoder-decoder model.

    T5 is much lighter (220M vs 3.8B) and well-suited for VQA tasks.
    No LoRA needed due to smaller size. Hidden dim (768) matches BioMedCLIP.
    """

    def __init__(self, encoder_dim: int = 768):
        super().__init__()
        
        self.model = T5ForConditionalGeneration.from_pretrained(
            "t5-base",
            torch_dtype=torch.float32
        )
        self.hidden_size = self.model.config.d_model  # 768

        # Optional projection if dims don't match (currently 768→768, so skip)
        self.input_proj = None
        if encoder_dim != self.hidden_size:
            self.input_proj = nn.Linear(encoder_dim, self.hidden_size)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        # fused: (B, N+L, encoder_dim)
        inputs_embeds = fused
        if self.input_proj is not None:
            inputs_embeds = self.input_proj(fused)  # (B, N+L, 768)
        
        batch_size = inputs_embeds.size(0)
        device = inputs_embeds.device
        
        # T5 needs decoder_input_ids; use pad token (0) to start generation
        decoder_input_ids = torch.zeros(
            batch_size, 1, dtype=torch.long, device=device
        )
        
        # Run T5 encoder-decoder
        outputs = self.model(
            inputs_embeds=inputs_embeds,          # Encoder input
            decoder_input_ids=decoder_input_ids,  # Decoder input (just pad)
            output_hidden_states=True,
            return_dict=True
        )
        
        # Return last decoder hidden state: (B, 1, 768)
        # Expand to (B, seq_len, 768) for compatibility with heads
        decoder_hidden = outputs.decoder_hidden_states[-1]  # (B, 1, 768)
        seq_len = inputs_embeds.size(1)
        
        # Repeat decoder output to match original sequence length
        return decoder_hidden.expand(-1, seq_len, -1)  # (B, N+L, 768)


class MedicalVQAModel(nn.Module):

    def __init__(
        self,
        encoder_dim: int = 768,
        vocab_size: int = 30522,
        max_answer_len: int = 16
    ):
        super().__init__()
        self.max_answer_len = max_answer_len

        self.encoder     = BioMedCLIPEncoder()
        self.dual_gating = DualGatingModule(encoder_dim)
        self.decoder     = T5Decoder(encoder_dim)

        dec_dim = self.decoder.hidden_size  # 768 (same as BioMedCLIP)

        # Yes/No classification head
        self.yesno_head = nn.Linear(dec_dim, 1)

        # Generative head — outputs in BioMedCLIP/PubMedBERT vocabulary
        self.gen_head   = nn.Linear(dec_dim, vocab_size)

    def forward(self, images, input_ids, attention_mask):
        # ── Encoder (frozen BioMedCLIP) ─────────────────────────────
        V, Q = self.encoder(images, input_ids, attention_mask)
        # V: (B, N, 768)   Q: (B, L, 768)

        # ── Question-Aware Dual Gating ──────────────────────────────
        V, Q = self.dual_gating(V, Q)

        # ── Feature Fusion ──────────────────────────────────────────
        fused = torch.cat([V, Q], dim=1)         # (B, N+L, 768)

        # ── Decoder (T5-base) ───────────────────────────────────────
        hidden = self.decoder(fused)             # (B, N+L, 768)

        # ── Output Heads ────────────────────────────────────────────
        cls          = hidden[:, 0]                              # (B, 768)
        yesno_logits = self.yesno_head(cls)                      # (B, 1)

        gen_hidden   = hidden[:, -self.max_answer_len:, :]       # (B, 16, 768)
        gen_logits   = self.gen_head(gen_hidden)                 # (B, 16, vocab_size)

        return yesno_logits, gen_logits