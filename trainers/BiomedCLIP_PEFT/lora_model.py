import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from .block_utils import vit_block_forward_with_lora


class MultiLayerLoRA(nn.Module):
    """LoRA wrapper managing low-rank QKV decomposition across selected ViT blocks."""

    def __init__(
        self,
        vision_dim: int = 768,
        rank: int = 4,
        layer_idxs: List[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if layer_idxs is None:
            layer_idxs = [3, 6, 9]
        self.layer_idxs = layer_idxs
        self.rank = rank
        self.vision_dim = vision_dim
        self.dropout = dropout

        self.lora_params = nn.ParameterDict({
            f'layer_{idx}': nn.ParameterDict({
                'q_A': nn.Parameter(torch.randn(rank, vision_dim) * 0.01),
                'q_B': nn.Parameter(torch.zeros(vision_dim, rank)),
                'k_A': nn.Parameter(torch.randn(rank, vision_dim) * 0.01),
                'k_B': nn.Parameter(torch.zeros(vision_dim, rank)),
                'v_A': nn.Parameter(torch.randn(rank, vision_dim) * 0.01),
                'v_B': nn.Parameter(torch.zeros(vision_dim, rank)),
            })
            for idx in layer_idxs
        })

        print(f"  [MultiLayerLoRA] vision_dim={vision_dim}, rank={rank}, "
              f"layers={layer_idxs}, dropout={dropout}")

    def get_params(self, layer_idx: int) -> Dict[str, nn.Parameter]:
        return self.lora_params[f'layer_{layer_idx}']


class BiomedCLIP_LoRA_Model(nn.Module):
    """LoRA-enhanced BiomedCLIP: frozen backbone + trainable LoRA QKV updates."""

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

        lora_cfg = cfg.TRAINER.BIOMEDCLIP_LORA.LORA
        self.rank = lora_cfg.RANK
        self.alpha = lora_cfg.ALPHA
        self.layer_idxs = lora_cfg.LAYER_IDXS

        self.lora_wrapper = MultiLayerLoRA(
            vision_dim=768, rank=self.rank,
            layer_idxs=self.layer_idxs, dropout=lora_cfg.get('DROPOUT', 0.0),
        )

        print(f"[BiomedCLIP_LoRA_Model] Visual embed_dim: {self.embed_dim}")
        print(f"[BiomedCLIP_LoRA_Model] Num classes: {self.num_classes}")
        print(f"[BiomedCLIP_LoRA_Model] LoRA rank: {self.rank}, alpha: {self.alpha}")

    def get_text_embeddings(self, classnames=None):
        classnames = self.classnames
        texts = [f"a photo of {name}" for name in classnames]
        text_tokens = self.tokenizer(texts).to(next(self.parameters()).device)
        with torch.no_grad():
            text_embeds = self.text_encoder(text_tokens)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def _forward_with_lora(self, image: torch.Tensor) -> torch.Tensor:
        """ViT forward with LoRA QKV injection at target blocks."""
        device = image.device
        dtype = self.dtype
        batch_size = image.shape[0]
        target_layers = set(self.layer_idxs)

        x = self.visual.trunk.patch_embed(image.type(dtype))
        cls_token = self.visual.trunk.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.visual.trunk.pos_embed
        x = self.visual.trunk.pos_drop(x)
        x = self.visual.trunk.norm_pre(x)

        for block_idx, block in enumerate(self.visual.trunk.blocks):
            drop_path_rate = getattr(block, 'drop_path_rate', 0.0)

            if block_idx in target_layers:
                lora_params = self.lora_wrapper.get_params(block_idx)
                lora_params = dict(lora_params)
                lora_params['alpha'] = torch.tensor(self.alpha, device=device)
                lora_params['dropout'] = torch.tensor(self.lora_wrapper.dropout, device=device)
                x = vit_block_forward_with_lora(block, x, lora_params, drop_path_rate, device)
            else:
                x = block(x)

        x = self.visual.trunk.norm(x)
        cls_feature = x[:, 0]
        adapted_image_feature = self.visual.head(cls_feature)
        adapted_image_feature = F.normalize(adapted_image_feature, dim=-1)
        return adapted_image_feature

    def forward(self, image: torch.Tensor, label=None) -> dict:
        visual_features = self._forward_with_lora(image)
        text_features = self.get_text_embeddings()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * visual_features @ text_features.t()

        output = {"logits": logits, "visual_features": visual_features}
        if label is not None:
            output["loss"] = F.cross_entropy(logits, label)
        return output
