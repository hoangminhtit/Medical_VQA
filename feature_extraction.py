import torch
import torch.nn as nn
from transformers import AutoModel

class BioMedCLIPEncoder(nn.Module):
    def __init__(self,
                model = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"    
            ):
        super().__init__()

        self.model = AutoModel.from_pretrained(
            model
        )

        # freeze encoder
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, images, input_ids, attention_mask):

        outputs = self.model(
            pixel_values=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        V = outputs.vision_model_output.last_hidden_state
        Q = outputs.text_model_output.last_hidden_state

        return V, Q