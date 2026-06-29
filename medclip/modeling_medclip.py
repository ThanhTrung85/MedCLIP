import pdb
import os
import copy
from collections import defaultdict
import requests

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torchvision

from . import constants


# =========================
# TEXT ENCODER
# =========================
class MedCLIPTextModel(nn.Module):
    def __init__(self,
        bert_type=constants.BERT_TYPE,
        proj_dim=512,
        proj_bias=False) -> None:

        super().__init__()
        self.bert_type = bert_type
        self.last_n_layer = 4

        self.model = AutoModel.from_pretrained(
            self.bert_type,
            output_hidden_states=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.projection_head = nn.Linear(768, proj_dim, bias=proj_bias)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # ===== FIX 1: HF compatibility (ensure hidden_states exists) =====
        hidden_states = output.hidden_states

        last_hidden_states = torch.stack([
            hidden_states[1],
            hidden_states[2],
            hidden_states[-1]
        ])

        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)

        embed = self.projection_head(embed)
        return embed


# =========================
# RESNET VISION ENCODER
# =========================
class MedCLIPVisionModel(nn.Module):
    def __init__(self, checkpoint=None, medclip_checkpoint=None):
        super().__init__()

        # ===== FIX 2: torchvision new API (deprecated pretrained=True) =====
        self.model = torchvision.models.resnet50(weights="IMAGENET1K_V1")

        num_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fts, 512, bias=False)

        if checkpoint is not None:
            # ===== FIX 3: avoid CUDA mismatch =====
            state_dict = torch.load(
                os.path.join(checkpoint, constants.WEIGHTS_NAME),
                map_location="cpu"
            )
            self.load_state_dict(state_dict, strict=False)

        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        state_dict = torch.load(
            os.path.join(checkpoint, constants.WEIGHTS_NAME),
            map_location="cpu"
        )

        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.', '')] = state_dict[key]

        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, pixel_values, **kwargs):
        # ===== FIX 4: safe channel handling =====
        if pixel_values.shape[1] == 1:
            pixel_values = pixel_values.repeat((1, 3, 1, 1))

        return self.model(pixel_values)


# =========================
# VIT VISION ENCODER
# =========================
class MedCLIPVisionModelViT(nn.Module):
    def __init__(self, checkpoint=None, medclip_checkpoint=None) -> None:
        super().__init__()

        self.vit_type = constants.VIT_TYPE
        self.model = AutoModel.from_pretrained(self.vit_type)
        self.projection_head = nn.Linear(768, 512, bias=False)

        if checkpoint is not None:
            state_dict = torch.load(
                os.path.join(checkpoint, constants.WEIGHTS_NAME),
                map_location="cpu"
            )
            self.load_state_dict(state_dict, strict=False)

    def forward(self, pixel_values, project=True):
        if pixel_values.shape[1] == 1:
            pixel_values = pixel_values.repeat((1, 3, 1, 1))

        output = self.model(pixel_values)

        # ===== FIX 5: HF ViT output compatibility =====
        if hasattr(output, "pooler_output"):
            img_embeds = output.pooler_output
        else:
            img_embeds = output.last_hidden_state[:, 0]

        if project:
            img_embeds = self.projection_head(img_embeds)

        return img_embeds


# =========================
# MAIN MEDCLIP MODEL
# =========================
class MedCLIPModel(nn.Module):
    def __init__(self,
        vision_cls=MedCLIPVisionModel,
        checkpoint=None,
        vision_checkpoint=None,
        logit_scale_init_value=0.07):

        super().__init__()

        assert vision_cls in [MedCLIPVisionModel, MedCLIPVisionModelViT]

        self.vision_model = vision_cls(checkpoint=vision_checkpoint)
        self.text_model = MedCLIPTextModel(proj_bias=False)

        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1 / logit_scale_init_value))
        )

        if checkpoint is not None:
            state_dict = torch.load(
                os.path.join(checkpoint, constants.WEIGHTS_NAME),
                map_location="cpu"
            )
            self.load_state_dict(state_dict, strict=False)

    # =========================
    # FIX 6: remove implicit .cuda()
    # =========================
    def encode_text(self, input_ids=None, attention_mask=None):
        device = input_ids.device

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        text_embeds = self.text_model(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def encode_image(self, pixel_values=None):
        pixel_values = pixel_values.to(pixel_values.device)
        img_embeds = self.vision_model(pixel_values)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        return img_embeds

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()

        logits = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits.t()

    def clip_loss(self, similarity):
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits):
        return nn.functional.cross_entropy(
            logits,
            torch.arange(len(logits), device=logits.device)
        )

    def forward(self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        return_loss=None,
        **kwargs):

        # ===== FIX 7: safe device sync (minimal change) =====
        device = pixel_values.device

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        pixel_values = pixel_values.to(device)

        img_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        logits_per_image = self.compute_logits(img_embeds, text_embeds)
        logits_per_text = logits_per_image.t()

        loss = self.clip_loss(logits_per_text) if return_loss else None

        return {
            'img_embeds': img_embeds,
            'text_embeds': text_embeds,
            'logits': logits_per_image,
            'loss_value': loss,
            'logits_per_text': logits_per_text
        }


# =========================
# ZERO-SHOT CLASSIFIER
# =========================
class PromptClassifier(nn.Module):
    def __init__(self, medclip_model, ensemble=False):
        super().__init__()
        self.model = medclip_model
        self.ensemble = ensemble

    def forward(self, pixel_values=None, prompt_inputs=None):

        # ===== FIX 8: remove hard cuda() =====
        device = pixel_values.device
        pixel_values = pixel_values.to(device)

        class_similarities = []
        class_names = []

        for cls_name, cls_text in prompt_inputs.items():

            inputs = {"pixel_values": pixel_values}

            for k in cls_text:
                inputs[k] = cls_text[k].to(device)

            outputs = self.model(**inputs)
            logits = outputs['logits']

            cls_sim = logits.mean(1) if self.ensemble else logits.max(1)[0]

            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        return {
            'logits': torch.stack(class_similarities, 1),
            'class_names': class_names
        }


# =========================
# SUPERVISED CLASSIFIER
# =========================
class SuperviseClassifier(nn.Module):
    def __init__(self,
        vision_model,
        num_class=14,
        input_dim=768,
        mode=None):

        super().__init__()

        self.model = vision_model
        self.num_class = num_class
        self.mode = mode.lower()

        assert self.mode in ['multiclass', 'multilabel', 'binary']

        if num_class > 2:
            self.fc = nn.Linear(input_dim, num_class)
            self.loss_fn = nn.CrossEntropyLoss() if self.mode == 'multiclass' else nn.BCEWithLogitsLoss()
        else:
            self.fc = nn.Linear(input_dim, 1)
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self,
        pixel_values,
        labels=None,
        return_loss=True):

        # ===== FIX 9: device-safe =====
        device = pixel_values.device
        pixel_values = pixel_values.to(device)

        img_embeds = self.model(pixel_values, project=False)
        logits = self.fc(img_embeds)

        out = {
            'embedding': img_embeds,
            'logits': logits
        }

        if labels is not None and return_loss:
            labels = labels.to(device).float()

            if self.mode == 'multiclass':
                labels = labels.long().view(-1)

            out['loss_value'] = self.loss_fn(logits, labels)

        return out