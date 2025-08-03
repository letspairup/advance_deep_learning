import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor


class CLIP(nn.Module):
    def __init__(self, vision_model="openai/clip-vit-base-patch32", text_model="openai/clip-vit-base-patch32", proj_dim=256, temperature=0.07):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model)
        self.text_encoder = CLIPTextModel.from_pretrained(text_model)

        self.image_proj = nn.Linear(self.vision_encoder.config.hidden_size, proj_dim, bias=False)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, proj_dim, bias=False)

        # logit_scale initialized to log(1/temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / temperature)))

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None, **kwargs):
        vision_outputs = self.vision_encoder(pixel_values=pixel_values).last_hidden_state[:, 0]  # CLS token
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

        vision_embed = nn.functional.normalize(self.image_proj(vision_outputs), dim=-1)
        text_embed = nn.functional.normalize(self.text_proj(text_outputs), dim=-1)

        logit_scale = self.logit_scale.exp()

        return vision_embed, text_embed, logit_scale


def compute_clip_loss(outputs, labels, num_items_in_batch=None):
    image_embeds, text_embeds, logit_scale = outputs
    logits_per_image = logit_scale * image_embeds @ text_embeds.T
    logits_per_text = logits_per_image.T

    batch_size = image_embeds.size(0)
    target = torch.arange(batch_size).to(image_embeds.device)

    loss_i2t = nn.functional.cross_entropy(logits_per_image, target)
    loss_t2i = nn.functional.cross_entropy(logits_per_text, target)

    return (loss_i2t + loss_t2i) / 2
