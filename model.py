import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

from feature_extraction import ImageEncoder, TextEncoder
from dual_gating_attention import DualGatingModule


# ──────────────────────────────────────────────────────────────────────────────
# Fusion Layer: concat([V', Q']) → Linear → LayerNorm
# ──────────────────────────────────────────────────────────────────────────────

class FusionLayer(nn.Module):
    """Fuses image and text tokens into a single sequence.

    Architecture: concat([V', Q']) → Linear → LayerNorm
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear   = nn.Linear(dim * 2, dim)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, V: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Fuses token-by-token for each position independently.
        For VQA we concatenate along the sequence dimension first,
        then apply a token-wise projection.

        Args:
            V: (B, Nv, dim)
            Q: (B, Nq, dim)
        Returns:
            fused: (B, Nv + Nq, dim)
        """
        # Simple sequence concat — the Linear+LN operate token-wise
        fused = torch.cat([V, Q], dim=1)            # (B, Nv+Nq, 2*dim) — wait, dim is same
        # Correct: both V and Q have shape (B, N, dim), concat on seq dim → (B, Nv+Nq, dim)
        # The "Linear → LayerNorm" in the architecture refers to a post-concat projection.
        # Since we concat along seq dim (not feature dim), we do a token-wise MLP:
        fused = self.layernorm(fused)  # (B, Nv+Nq, dim) — apply LN token-wise
        return fused


class FusionProjection(nn.Module):
    """Projects fused [V; Q] sequence with a linear + layer norm.

    This matches the architecture spec:
        concat([V', Q']) → Linear → LayerNorm
    where the Linear reduces channel dim if needed, or acts as a mix.
    Since V and Q both have `dim` features, we use a dim→dim projection.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, V: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            V: (B, Nv, dim)
            Q: (B, Nq, dim)
        Returns:
            fused: (B, Nv+Nq, dim)
        """
        fused = torch.cat([V, Q], dim=1)   # (B, Nv+Nq, dim)
        fused = self.proj(fused)           # (B, Nv+Nq, dim)
        fused = self.norm(fused)           # (B, Nv+Nq, dim)
        fused = self.drop(fused)
        return fused


# ──────────────────────────────────────────────────────────────────────────────
# Yes/No Head: 2-layer MLP with BCE Loss
# ──────────────────────────────────────────────────────────────────────────────

class YesNoHead(nn.Module):
    """2-layer MLP for binary Yes/No classification.

    Architecture: Linear → ReLU → Dropout → Linear → (1 logit for BCE)
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 1)


# ──────────────────────────────────────────────────────────────────────────────
# T5-small Generative Decoder
# ──────────────────────────────────────────────────────────────────────────────

class T5Decoder(nn.Module):
    """T5-small encoder-decoder model for VQA text generation.

    T5-small is lighter and well-suited for VQA tasks.
    Uses proper autoregressive text generation instead of parallel prediction.
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        max_answer_len: int = 16,
        model_name: str = "t5-small",
        gen_num_beams: int = 4,
        gen_repetition_penalty: float = 1.2,
        gen_no_repeat_ngram_size: int = 2,
    ):
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        self.hidden_size = self.model.config.d_model  # 512 for T5-small
        self.max_answer_len = max_answer_len
        self.gen_num_beams = gen_num_beams
        self.gen_repetition_penalty = gen_repetition_penalty
        self.gen_no_repeat_ngram_size = gen_no_repeat_ngram_size

        # Project fused features → T5-small hidden dim (512)
        self.input_proj = nn.Linear(encoder_dim, self.hidden_size) \
            if encoder_dim != self.hidden_size else nn.Identity()

    def _prepare_inputs(self, fused: torch.Tensor) -> torch.Tensor:
        return self.input_proj(fused)

    def encode(
        self,
        fused: torch.Tensor,
        encoder_attention_mask: torch.Tensor = None
    ):
        """Run T5 encoder and return encoder_outputs (for reuse in eval)."""
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
        """Forward pass with dual mode: training vs generation.

        Args:
            fused:                  (B, N+L, encoder_dim)
            encoder_attention_mask: (B, N+L)
            labels:                 (B, max_answer_len) T5 token IDs (training only)
            generate_text:          If True, use autoregressive generation.
            encoder_outputs:        Pre-computed encoder outputs (eval shortcut).
        """
        inputs_embeds = self._prepare_inputs(fused)
        batch_size    = inputs_embeds.size(0)
        device        = inputs_embeds.device

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_attention_mask
            )

        if generate_text:
            # ── Autoregressive Text Generation ──────────────────────────
            generated_ids = self.model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_attention_mask,
                max_new_tokens=self.max_answer_len,
                num_beams=self.gen_num_beams,
                do_sample=False,
                repetition_penalty=self.gen_repetition_penalty,
                no_repeat_ngram_size=self.gen_no_repeat_ngram_size,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            return generated_ids.sequences  # (B, gen_len)

        else:
            if labels is not None:
                # ── Training Mode ────────────────────────────────────────
                decoder_input_ids = self.model._shift_right(labels)
                outputs = self.model(
                    encoder_outputs=encoder_outputs,
                    attention_mask=encoder_attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    output_hidden_states=True,
                    return_dict=True
                )
                # CLS token of last decoder layer → for Yes/No head input
                cls_hidden = outputs.decoder_hidden_states[-1][:, 0]  # (B, hidden_size)
                gen_logits = outputs.logits                             # (B, L, vocab_size)
                return cls_hidden, gen_logits

            else:
                # ── Classification-only (no labels) ─────────────────────
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
                cls_hidden = outputs.decoder_hidden_states[-1][:, 0]  # (B, hidden_size)
                return cls_hidden


# ──────────────────────────────────────────────────────────────────────────────
# Main Medical VQA Model
# ──────────────────────────────────────────────────────────────────────────────

class MedicalVQAModel(nn.Module):
    """Medical VQA model with BLIP-2 ViT-L + BioBERT + Dual Gating + T5-small.

    Architecture:
        Image  → BLIP-2 ViT-L (top 4 layers unfrozen) → (B, Nv, 768)
        Text   → BioBERT-base  (top 3 layers unfrozen) → (B, Nq, 768)
                       ↓
        Question-Aware Dual Gating (Pre-LN + Dropout)
                       ↓
        Fusion: concat([V', Q']) → Linear → LayerNorm → (B, Nv+Nq, 768)
                       ↓
          ┌────────────┴────────────┐
      Yes/No Head              T5-small Decoder
      (2-layer MLP)           (Generative Head)
      BCE Loss                 Cross-Entropy Loss
    """

    def __init__(
        self,
        dim: int = 768,
        max_answer_len: int = 16,
        image_unfreeze_top: int = 4,
        text_unfreeze_top: int = 3,
        t5_model: str = "t5-small",
        gen_num_beams: int = 4,
        gen_repetition_penalty: float = 1.2,
        gen_no_repeat_ngram_size: int = 2,
    ):
        super().__init__()
        self.max_answer_len = max_answer_len
        self.dim = dim

        # ── Encoders ────────────────────────────────────────────────────
        self.image_encoder = ImageEncoder(out_dim=dim, unfreeze_top=image_unfreeze_top)
        self.text_encoder  = TextEncoder(unfreeze_top=text_unfreeze_top)

        # ── Question-Aware Dual Gating ──────────────────────────────────
        self.dual_gating = DualGatingModule(dim)

        # ── Fusion Layer ────────────────────────────────────────────────
        self.fusion = FusionProjection(dim)

        # ── Generative Decoder (T5-small) ────────────────────────────────
        self.decoder = T5Decoder(
            encoder_dim=dim,
            max_answer_len=max_answer_len,
            model_name=t5_model,
            gen_num_beams=gen_num_beams,
            gen_repetition_penalty=gen_repetition_penalty,
            gen_no_repeat_ngram_size=gen_no_repeat_ngram_size,
        )
        dec_dim = self.decoder.hidden_size  # 512 for T5-small

        # ── Yes/No Head (2-layer MLP) ────────────────────────────────────
        self.yesno_head = YesNoHead(in_dim=dec_dim, hidden_dim=256)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _build_fused_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        # Encode image and text
        V = self.image_encoder(pixel_values)          # (B, Nv, dim)
        Q = self.text_encoder(input_ids, attention_mask)  # (B, Nq, dim)

        # Dual gating
        V, Q = self.dual_gating(V, Q)

        # Fusion: concat → Linear → LayerNorm
        fused = self.fusion(V, Q)                     # (B, Nv+Nq, dim)

        # Build attention mask for fused sequence
        v_mask    = torch.ones(
            V.size(0), V.size(1),
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        fused_mask = torch.cat([v_mask, attention_mask], dim=1)  # (B, Nv+Nq)

        return fused, fused_mask

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        generate_text: bool = False,
        eval_mode: bool = False,
    ):
        """
        Args:
            pixel_values:   (B, 3, 224, 224)
            input_ids:      (B, L) — BioBERT tokenized question
            attention_mask: (B, L)
            labels:         (B, max_answer_len) — T5 tokenized answer (training)
            generate_text:  True → autoregressive generation
            eval_mode:      True → run encoder once, decode both heads

        Returns:
            (yesno_logits, gen_logits | generated_ids | None)
        """
        fused, fused_mask = self._build_fused_features(pixel_values, input_ids, attention_mask)

        if eval_mode:
            # ── Eval: encode once, reuse for both heads ──────────────────
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
            # ── Inference generation mode ────────────────────────────────
            generated_ids = self.decoder(
                fused, encoder_attention_mask=fused_mask, generate_text=True
            )
            return None, generated_ids

        else:
            # ── Training / classification mode ───────────────────────────
            if labels is not None:
                cls_features, gen_logits = self.decoder(
                    fused,
                    encoder_attention_mask=fused_mask,
                    labels=labels,
                    generate_text=False
                )
                yesno_logits = self.yesno_head(cls_features)
                return yesno_logits, gen_logits
            else:
                cls_features = self.decoder(
                    fused,
                    encoder_attention_mask=fused_mask,
                    generate_text=False
                )
                yesno_logits = self.yesno_head(cls_features)
                return yesno_logits, None