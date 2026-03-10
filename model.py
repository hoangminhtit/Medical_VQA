import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from feature_extraction import BioMedCLIPEncoder
from dual_gating_attention import DualGatingModule


def _patch_dynamic_cache():
    """Restore DynamicCache.from_legacy_cache removed in transformers ≥ 4.44.

    The cached Phi-3 revision (f39ac1d) still calls this classmethod.
    We add it back as a no-op shim when it is missing.
    """
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "from_legacy_cache"):
            @classmethod
            def from_legacy_cache(cls, past_key_values=None):
                cache = cls()
                if past_key_values is not None:
                    for layer_idx, layer_past in enumerate(past_key_values):
                        cache.update(layer_past[0], layer_past[1], layer_idx)
                return cache
            DynamicCache.from_legacy_cache = from_legacy_cache
    except ImportError:
        pass


_patch_dynamic_cache()


def _load_phi3_base():
    """Load Phi-3 Mini with rope_scaling compatibility fix.

    Phi-3-mini-4k-instruct does NOT use LongRoPE.
    Some cached revisions on Kaggle/HF ship an incomplete rope_scaling dict
    (missing 'type', 'short_factor', 'long_factor'), causing KeyError at init.
    The safe fix is to nullify rope_scaling entirely when required keys are absent.
    """
    _CKPT = "microsoft/Phi-3-mini-4k-instruct"
    cfg = AutoConfig.from_pretrained(_CKPT, trust_remote_code=True)

    # If rope_scaling is present but incomplete (not a proper LongRoPE config),
    # remove it so the model falls back to standard RoPE.
    if isinstance(getattr(cfg, "rope_scaling", None), dict):
        required = {"type", "short_factor", "long_factor"}
        if not required.issubset(cfg.rope_scaling.keys()):
            cfg.rope_scaling = None

    return AutoModelForCausalLM.from_pretrained(
        _CKPT,
        config=cfg,
        trust_remote_code=True
    )


class PhiDecoder(nn.Module):
    """Phi-3 Mini decoder fine-tuned with LoRA.

    Projects BioMedCLIP encoder features (dim=768) into Phi-3's hidden space
    (3072), runs them through the LLM, and returns the last-layer hidden states.
    Only LoRA adapter weights and the input projection are trainable.
    """

    def __init__(self, encoder_dim: int = 768):
        super().__init__()

        base = _load_phi3_base()

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["qkv_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        self.model = get_peft_model(base, lora_cfg)
        self.hidden_size = self.model.config.hidden_size  # 3072

        # Bridge encoder dim (768) → decoder hidden dim (3072)
        self.input_proj = nn.Linear(encoder_dim, self.hidden_size)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        # fused: (B, N+L, encoder_dim)
        embeds = self.input_proj(fused)          # (B, N+L, hidden_size)
        out = self.model(
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True
        )
        return out.hidden_states[-1]             # (B, N+L, hidden_size)


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
        self.decoder     = PhiDecoder(encoder_dim)

        dec_dim = self.decoder.hidden_size  # 3072

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

        # ── Decoder (Phi-3 Mini + LoRA) ─────────────────────────────
        hidden = self.decoder(fused)             # (B, N+L, 3072)

        # ── Output Heads ────────────────────────────────────────────
        cls          = hidden[:, 0]                              # (B, 3072)
        yesno_logits = self.yesno_head(cls)                      # (B, 1)

        gen_hidden   = hidden[:, -self.max_answer_len:, :]       # (B, 16, 3072)
        gen_logits   = self.gen_head(gen_hidden)                 # (B, 16, vocab_size)

        return yesno_logits, gen_logits