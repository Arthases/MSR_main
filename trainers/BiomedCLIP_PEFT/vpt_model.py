import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class VPT_Model(nn.Module):
    """VPT-Deep: learnable prompt tokens injected at target ViT block inputs."""

    def __init__(self, cfg, classnames, biomedclip_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.num_classes = len(classnames)

        self.visual = biomedclip_model.visual
        self.dtype = biomedclip_model.text.transformer.dtype
        self.text_encoder = biomedclip_model.text
        self.logit_scale = biomedclip_model.logit_scale
        self.tokenizer = tokenizer

        dummy = torch.zeros(1, 3, 224, 224).to(next(self.visual.parameters()).device)
        with torch.no_grad():
            out = self.visual(dummy.type(self.dtype))
        self.embed_dim = out.shape[1]

        vpt_cfg = cfg.TRAINER.BIOMEDCLIP_VPT.VPT
        self.n_prompts = vpt_cfg.N_PROMPTS
        self.layer_idxs: List[int] = list(vpt_cfg.LAYER_IDXS)
        self.vit_dim = 768

        self.prompts = nn.ParameterDict()
        for idx in self.layer_idxs:
            prompt = nn.Parameter(torch.empty(1, self.n_prompts, self.vit_dim))
            nn.init.xavier_uniform_(prompt)
            self.prompts[f'layer_{idx}'] = prompt

        n_vpt_params = sum(p.numel() for p in self.prompts.parameters())
        print(f"[VPT_Model] n_prompts={self.n_prompts}, layers={self.layer_idxs}")
        print(f"[VPT_Model] VPT params: {n_vpt_params:,}")

    def get_text_embeddings(self, classnames=None):
        classnames = self.classnames
        texts = [f"a photo of {name}" for name in classnames]
        text_tokens = self.tokenizer(texts).to(next(self.parameters()).device)
        with torch.no_grad():
            text_embeds = self.text_encoder(text_tokens)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def _forward_with_vpt(self, image: torch.Tensor) -> torch.Tensor:
        """ViT forward: inject [CLS, prompts, patches] at target layers, remove after block."""
        device = image.device
        dtype = self.dtype
        batch_size = image.shape[0]
        target_layers = set(self.layer_idxs)

        x = self.visual.trunk.patch_embed(image.type(dtype))
        cls_token = self.visual.trunk.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.visual.trunk.pos_embed
        x = self.visual.trunk.pos_drop(x)
        x = self.visual.trunk.norm_pre(x) if hasattr(self.visual.trunk, 'norm_pre') else x

        for block_idx, block in enumerate(self.visual.trunk.blocks):
            if block_idx in target_layers:
                prompts = self.prompts[f'layer_{block_idx}'].expand(batch_size, -1, -1)
                x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
                x = block(x)
                x = torch.cat([x[:, :1], x[:, 1 + self.n_prompts:]], dim=1)
            else:
                x = block(x)

        x = self.visual.trunk.norm(x)
        cls_feature = x[:, 0]
        adapted_image_feature = self.visual.head(cls_feature)
        adapted_image_feature = F.normalize(adapted_image_feature, dim=-1)
        return adapted_image_feature

    def forward(self, image: torch.Tensor, label=None) -> dict:
        visual_features = self._forward_with_vpt(image)
        text_features = self.get_text_embeddings()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * visual_features @ text_features.t()

        output = {"logits": logits, "visual_features": visual_features}
        if label is not None:
            output["loss"] = F.cross_entropy(logits, label)
        return output
