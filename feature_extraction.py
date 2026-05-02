import torch
import torch.nn as nn
from transformers import (
    Blip2VisionModel,
    AutoModel,
    AutoTokenizer,
)


def _unfreeze_top_n_layers(module: nn.Module, layer_list_attr: str, n: int):
    """Freeze all layers first, then unfreeze the last `n` layers."""
    # Freeze everything first
    for p in module.parameters():
        p.requires_grad = False

    layers = getattr(module, layer_list_attr, None)
    if layers is None:
        return
    for layer in layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True


class ImageEncoder(nn.Module):
    """BLIP-2 ViT-L image encoder with top-4 layers unfrozen.

    BLIP-2 ViT-L outputs hidden states of dimension 1024.
    We project to `out_dim` (default 768) to match the text encoder.
    """
    _MODEL = "Salesforce/blip2-opt-2.7b"  # ViT-L is part of this checkpoint
    # Alternatively use "Salesforce/blip2-flan-t5-xl" — same ViT-L backbone

    def __init__(self, out_dim: int = 768, unfreeze_top: int = 4):
        super().__init__()

        self.vit = Blip2VisionModel.from_pretrained(
            self._MODEL,
            torch_dtype=torch.float32  # fp32 to avoid NaN gradients in mixed-precision
        )
        self.hidden_dim = self.vit.config.hidden_size  # 1408 for ViT-L

        # Freeze all, then unfreeze top `unfreeze_top` encoder layers
        for p in self.vit.parameters():
            p.requires_grad = False

        encoder_layers = self.vit.encoder.layers
        for layer in encoder_layers[-unfreeze_top:]:
            for p in layer.parameters():
                p.requires_grad = True
        # Also unfreeze final layernorm
        for p in self.vit.post_layernorm.parameters():
            p.requires_grad = True

        # Project ViT-L (1408) → out_dim (768)
        self.proj = nn.Linear(self.hidden_dim, out_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224)
        Returns:
            V: (B, num_patches, out_dim)
        """
        outputs = self.vit(pixel_values=pixel_values, return_dict=True)
        # last_hidden_state: (B, num_patches+1, 1408) — includes CLS token
        V = outputs.last_hidden_state  # already float32
        V = self.proj(V)   # (B, num_patches+1, out_dim)
        return V


class TextEncoder(nn.Module):
    """BioBERT-base text encoder with top-3 layers unfrozen.

    BioBERT outputs 768-dim hidden states — no projection needed.
    """
    _MODEL = "dmis-lab/biobert-base-cased-v1.2"

    def __init__(self, unfreeze_top: int = 3):
        super().__init__()

        self.bert = AutoModel.from_pretrained(self._MODEL)
        # Freeze all, then unfreeze top `unfreeze_top` transformer layers
        for p in self.bert.parameters():
            p.requires_grad = False

        encoder_layers = self.bert.encoder.layer
        for layer in encoder_layers[-unfreeze_top:]:
            for p in layer.parameters():
                p.requires_grad = True
        # Unfreeze pooler
        if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
            for p in self.bert.pooler.parameters():
                p.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)
        Returns:
            Q: (B, L, 768)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        Q = outputs.last_hidden_state.float()  # (B, L, 768)
        return Q