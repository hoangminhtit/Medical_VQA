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

    def _prepare_inputs(self, fused: torch.Tensor) -> torch.Tensor:
        inputs_embeds = fused
        if self.input_proj is not None:
            inputs_embeds = self.input_proj(fused)
        return inputs_embeds

    def encode(
        self,
        fused: torch.Tensor,
        encoder_attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        inputs_embeds = self._prepare_inputs(fused)
        return self.model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_attention_mask
        )

    def forward(
        self,
        fused: torch.Tensor,
        encoder_attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        generate_text: bool = False,
        encoder_outputs=None
    ) -> torch.Tensor:
        """Forward pass with dual mode: classification vs generation.
        
        Args:
            fused: (B, N+L, encoder_dim) - concatenated vision + text features
            labels: (B, max_answer_len) - target token IDs for training (T5 tokenized)
            generate_text: If True, use autoregressive generation for text.
                          If False, return pooled features for classification.
        """
        inputs_embeds = self._prepare_inputs(fused)  # (B, N+L, 768)

        batch_size = inputs_embeds.size(0)
        device = inputs_embeds.device

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_attention_mask
            )
        
        if generate_text:
            # ── Autoregressive Text Generation ──────────────────────────
            # Use T5's generate() method for proper language modeling
            # Generate text autoregressively
            generated_ids = self.model.generate(
                encoder_outputs=encoder_outputs,
                encoder_attention_mask=encoder_attention_mask,
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
                    encoder_outputs=encoder_outputs,
                    attention_mask=encoder_attention_mask,
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
                    encoder_outputs=encoder_outputs,
                    attention_mask=encoder_attention_mask,
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
        fused = torch.cat([V, Q], dim=1)         # (B, N+L, 768)

        # Build encoder attention mask for fused tokens.
        v_mask = torch.ones(
            V.size(0), V.size(1),
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        fused_mask = torch.cat([v_mask, attention_mask], dim=1)
        return fused, fused_mask

    def forward(self, images, input_ids, attention_mask, labels=None, generate_text: bool = False):
        fused, fused_mask = self._build_fused_features(images, input_ids, attention_mask)

        if generate_text is None:
            # Evaluation mode: run encoder/gating once, then decode both heads.
            encoder_outputs = self.decoder.encode(fused, encoder_attention_mask=fused_mask)
            cls_features = self.decoder(
                fused,
                encoder_attention_mask=fused_mask,
                generate_text=False,
                encoder_outputs=encoder_outputs
            )
            yesno_logits = self.yesno_head(cls_features)
            generated_ids = self.decoder(
                fused,
                encoder_attention_mask=fused_mask,
                generate_text=True,
                encoder_outputs=encoder_outputs
            )
            return yesno_logits, generated_ids

        if generate_text:
            # ── Text Generation Mode ────────────────────────────────
            # Use T5's autoregressive generation
            generated_ids = self.decoder(
                fused, encoder_attention_mask=fused_mask, generate_text=True
            )  # (B, gen_len)
            return None, generated_ids  # Return (None, generated_ids)
        
        else:
            # ── Training/Classification Mode ────────────────────────
            if labels is not None:
                # Training mode: get both classification features and generation logits
                cls_features, gen_logits = self.decoder(
                    fused, encoder_attention_mask=fused_mask, labels=labels, generate_text=False
                )
                yesno_logits = self.yesno_head(cls_features)              # (B, 1)
                return yesno_logits, gen_logits
            else:
                # Classification-only mode 
                cls_features = self.decoder(
                    fused, encoder_attention_mask=fused_mask, generate_text=False
                )   # (B, 768)
                yesno_logits = self.yesno_head(cls_features)              # (B, 1)

                return yesno_logits, None