import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

from feature_extraction import BioMedCLIPEncoder
from dual_gating_attention import DualGatingModule


class T5Decoder(nn.Module):
    """T5-base encoder-decoder model for VQA text generation.

    T5 is much lighter (220M vs 3.8B) and well-suited for VQA tasks.
    Uses proper autoregressive text generation instead of parallel prediction.
    """

    def __init__(self, encoder_dim: int = 768, max_answer_len: int = 16):
        super().__init__()
        
        self.model = T5ForConditionalGeneration.from_pretrained(
            "t5-base",
            torch_dtype=torch.float32
        )
        self.hidden_size = self.model.config.d_model  # 768
        self.max_answer_len = max_answer_len

        # Optional projection if dims don't match
        self.input_proj = None
        if encoder_dim != self.hidden_size:
            self.input_proj = nn.Linear(encoder_dim, self.hidden_size)

    def forward(self, fused: torch.Tensor, generate_text: bool = False) -> torch.Tensor:
        """Forward pass with dual mode: classification vs generation.
        
        Args:
            fused: (B, N+L, encoder_dim) - concatenated vision + text features
            generate_text: If True, use autoregressive generation for text.
                          If False, return pooled features for classification.
        """
        inputs_embeds = fused
        if self.input_proj is not None:
            inputs_embeds = self.input_proj(fused)  # (B, N+L, 768)
        
        batch_size = inputs_embeds.size(0)
        device = inputs_embeds.device
        
        if generate_text:
            # ── Autoregressive Text Generation ──────────────────────────
            # Use T5's generate() method for proper language modeling
            encoder_outputs = self.model.encoder(inputs_embeds=inputs_embeds)
            
            # Generate text autoregressively
            generated_ids = self.model.generate(
                encoder_outputs=encoder_outputs,
                max_new_tokens=self.max_answer_len,
                num_beams=1,           # Greedy decoding for speed
                do_sample=False,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            
            # Return generated token IDs: (B, generated_len)
            return generated_ids.sequences
            
        else:
            # ── Classification Mode ─────────────────────────────────────
            # Single forward pass for pooled representation
            decoder_input_ids = torch.zeros(
                batch_size, 1, dtype=torch.long, device=device
            )
            
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Return pooled representation for classification
            cls_hidden = outputs.decoder_hidden_states[-1][:, 0]  # (B, 768)
            return cls_hidden


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
        self.decoder     = T5Decoder(encoder_dim, max_answer_len)

        dec_dim = self.decoder.hidden_size  # 768 (same as BioMedCLIP)

        # Yes/No classification head
        self.yesno_head = nn.Linear(dec_dim, 1)

        # Note: No gen_head needed - T5 has built-in language modeling head

    def forward(self, images, input_ids, attention_mask, generate_text: bool = False):
        # ── Encoder (frozen BioMedCLIP) ─────────────────────────────
        V, Q = self.encoder(images, input_ids, attention_mask)
        # V: (B, N, 768)   Q: (B, L, 768)

        # ── Question-Aware Dual Gating ──────────────────────────────
        V, Q = self.dual_gating(V, Q)

        # ── Feature Fusion ──────────────────────────────────────────
        fused = torch.cat([V, Q], dim=1)         # (B, N+L, 768)

        if generate_text:
            # ── Text Generation Mode ────────────────────────────────
            # Use T5's autoregressive generation
            generated_ids = self.decoder(fused, generate_text=True)  # (B, gen_len)
            return None, generated_ids  # Return (None, generated_ids)
        
        else:
            # ── Training/Classification Mode ────────────────────────
            cls_features = self.decoder(fused, generate_text=False)   # (B, 768)
            yesno_logits = self.yesno_head(cls_features)              # (B, 1)
            
            # For training, we still need gen_logits shape for loss computation
            # Create dummy logits - T5 will handle text generation separately
            batch_size = cls_features.size(0)
            dummy_gen_logits = torch.zeros(
                batch_size, self.max_answer_len, 32128,  # T5 vocab_size = 32128
                device=cls_features.device, dtype=cls_features.dtype
            )
            
            return yesno_logits, dummy_gen_logits