

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer
from ..prompt_templates import BIOMEDCOOP_TEMPLATES, CUSTOM_TEMPLATES


class ZeroShotBiomedCLIPModel(nn.Module):
    """Frozen BiomedCLIP model with cached text prototypes."""

    def __init__(self, biomedclip_model, text_features: torch.Tensor):
        super().__init__()
        self.visual = biomedclip_model.visual
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype
        self.register_buffer("text_features", text_features, persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self.visual(image.type(self.dtype))
        image_features = F.normalize(image_features, dim=-1)
        logit_scale = self.logit_scale.exp()
        return logit_scale * image_features @ self.text_features.t()


@TRAINER_REGISTRY.register()
class BiomedCLIP_ZeroShot(TrainerX):


    def check_cfg(self, cfg):
        zs_cfg = cfg.TRAINER.BIOMEDCLIP_ZEROSHOT
        assert zs_cfg.PREC in ["fp16", "fp32", "amp"]
        assert zs_cfg.PROMPT_MODE in ["custom", "ensemble"]

    def build_model(self):
        cfg = self.cfg
        zs_cfg = cfg.TRAINER.BIOMEDCLIP_ZEROSHOT
        classnames = self.dm.dataset.classnames

        print(f"\n{'='*60}")
        print(f">> Zero-shot BiomedCLIP: {cfg.DATASET.NAME}  classes={classnames}")
        print(f"   prompt_mode={zs_cfg.PROMPT_MODE}")
        print(f"{'='*60}")

        print("\n[1/3] Loading BiomedCLIP backbone...")
        biomedclip_model, _ = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            cache_dir="clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )

        if zs_cfg.PREC in ["fp32", "amp"]:
            biomedclip_model = biomedclip_model.float()

        tokenizer = get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            cache_dir="clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )

        biomedclip_model.to(self.device).eval()
        for param in biomedclip_model.parameters():
            param.requires_grad_(False)

        print("\n[2/3] Building zero-shot text prototypes...")
        text_features = self._build_text_features(
            cfg=cfg,
            model=biomedclip_model,
            tokenizer=tokenizer,
            classnames=classnames,
        )

        print("\n[3/3] Building frozen zero-shot model...")
        self.model = ZeroShotBiomedCLIPModel(
            biomedclip_model=biomedclip_model,
            text_features=text_features,
        ).to(self.device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_params:,} / {total_params:,}")
        print(f"Text prototypes: {tuple(text_features.shape)}")
        print(f"{'='*60}\n")

    def _build_text_features(self, cfg, model, tokenizer, classnames: List[str]) -> torch.Tensor:
        prompts_per_class = self._build_prompts(cfg, classnames)
        all_features = []

        with torch.no_grad():
            for classname, prompts in zip(classnames, prompts_per_class):
                tokenized = torch.cat([tokenizer(prompt) for prompt in prompts]).to(self.device)
                text_features = model.encode_text(tokenized)
                text_features = F.normalize(text_features, dim=-1)
                class_feature = text_features.mean(dim=0)
                class_feature = F.normalize(class_feature, dim=0)
                all_features.append(class_feature)
                print(f"  {classname}: {prompts}")

        return torch.stack(all_features, dim=0)

    def _build_prompts(self, cfg, classnames: List[str]) -> List[List[str]]:
        zs_cfg = cfg.TRAINER.BIOMEDCLIP_ZEROSHOT
        clean_classnames = [name.replace("_", " ") for name in classnames]

        if zs_cfg.PROMPT_MODE == "ensemble":
            prompts = []
            for original_name, clean_name in zip(classnames, clean_classnames):
                if original_name in BIOMEDCOOP_TEMPLATES:
                    prompts.append(BIOMEDCOOP_TEMPLATES[original_name])
                elif clean_name in BIOMEDCOOP_TEMPLATES:
                    prompts.append(BIOMEDCOOP_TEMPLATES[clean_name])
                else:
                    template = self._get_custom_template(cfg)
                    prompts.append([template.format(clean_name)])
            return prompts

        template = self._get_custom_template(cfg)
        return [[template.format(name)] for name in clean_classnames]

    def _get_custom_template(self, cfg) -> str:
        custom_template = cfg.TRAINER.BIOMEDCLIP_ZEROSHOT.CUSTOM_TEMPLATE
        if custom_template:
            return self._ensure_template_placeholder(custom_template)
        if cfg.DATASET.NAME in CUSTOM_TEMPLATES:
            return self._ensure_template_placeholder(CUSTOM_TEMPLATES[cfg.DATASET.NAME])
        return "a photo of a {}."

    def _ensure_template_placeholder(self, template: str) -> str:
        template = template.strip()
        if "{}" in template:
            return template
        template = template.rstrip(".:;,")
        return f"{template} {{}}."

    def model_inference(self, input):
        return self.model(input)

    def forward_backward(self, batch):
        raise RuntimeError(
            "BiomedCLIP_ZeroShot is an evaluation-only trainer. "
            "Please run it with --eval-only."
        )

    def load_model(self, directory, epoch=None):
        print("Zero-shot BiomedCLIP has no trainable checkpoint; load_model() is skipped.")
