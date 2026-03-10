import torch
import torch.nn as nn
import open_clip


class BioMedCLIPEncoder(nn.Module):
    # BioMedCLIP is an OpenCLIP model — must be loaded via open_clip,
    # NOT transformers.AutoModel (no model_type in its config.json).
    _MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(self):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms(self._MODEL)

        # Freeze the entire encoder
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images, input_ids, attention_mask):
        # Vision encoder: timm ViT-B/16
        # forward_features returns all tokens incl. CLS: (B, 197, 768)
        V = self.clip.visual.trunk.forward_features(images).float()

        # Text encoder: HFTextEncoder wrapping PubMedBERT
        # .transformer is the underlying HuggingFace BERT model
        text_out = self.clip.text.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        Q = text_out.last_hidden_state.float()  # (B, L, 768)

        return V, Q