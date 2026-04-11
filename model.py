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

    def forward(self, fused: torch.Tensor, labels: torch.Tensor = None, generate_text: bool = False) -> torch.Tensor:
        """Forward pass with dual mode: classification vs generation.
        
        Args:
            fused: (B, N+L, encoder_dim) - concatenated vision + text features
            labels: (B, max_answer_len) - target token IDs for training (T5 tokenized)
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
                num_beams=2,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,  # Avoid 2-gram repetition
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            
            # Return generated token IDs: (B, generated_len)
            return generated_ids.sequences
            
        else:
            if labels is not None:
                # ── Training Mode: Compute generation logits ─────────────
                # Prepare decoder input (shift right for T5)
                decoder_input_ids = self.model._shift_right(labels)
                
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,  # T5 will compute language modeling loss internally
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Return both classification features and generation logits
                cls_hidden = outputs.decoder_hidden_states[-1][:, 0]  # (B, 768)
                gen_logits = outputs.logits  # (B, max_answer_len, vocab_size)
                
                return cls_hidden, gen_logits
            else:
                # ── Classification Mode (without labels) ─────────────────
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
        vocab_size: int = 32128,
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

    def _build_fused_features(self, images, input_ids, attention_mask):
        # ── Encoder (frozen BioMedCLIP) ─────────────────────────────
        V, Q = self.encoder(images, input_ids, attention_mask)
        # V: (B, N, 768)   Q: (B, L, 768)

        # ── Question-Aware Dual Gating ──────────────────────────────
        V, Q = self.dual_gating(V, Q)

        # ── Feature Fusion ──────────────────────────────────────────
        return torch.cat([V, Q], dim=1)         # (B, N+L, 768)

    def forward(self, images, input_ids, attention_mask, labels=None, generate_text: bool = False):
        fused = self._build_fused_features(images, input_ids, attention_mask)

        if generate_text is None:
            # Evaluation mode: run encoder/gating once, then decode both heads.
            cls_features = self.decoder(fused, generate_text=False)
            yesno_logits = self.yesno_head(cls_features)
            generated_ids = self.decoder(fused, generate_text=True)
            return yesno_logits, generated_ids

        if generate_text:
            # ── Text Generation Mode ────────────────────────────────
            # Use T5's autoregressive generation
            generated_ids = self.decoder(fused, generate_text=True)  # (B, gen_len)
            return None, generated_ids  # Return (None, generated_ids)
        
        else:
            # ── Training/Classification Mode ────────────────────────
            if labels is not None:
                # Training mode: get both classification features and generation logits
                cls_features, gen_logits = self.decoder(fused, labels=labels, generate_text=False)
                yesno_logits = self.yesno_head(cls_features)              # (B, 1)
                return yesno_logits, gen_logits
            else:
                # Classification-only mode 
                cls_features = self.decoder(fused, generate_text=False)   # (B, 768)
                yesno_logits = self.yesno_head(cls_features)              # (B, 1)

                return yesno_logits, None