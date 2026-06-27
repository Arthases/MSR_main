import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .block_utils import vit_block_forward_with_ia3


class MultiLayerIA3(nn.Module):
    """Channel-wise scaling vectors for K, V, FFN at selected ViT blocks."""

    def __init__(self, vision_dim: int = 768, ffn_dim: int = 3072, layer_idxs: List[int] = None):
        super().__init__()
        if layer_idxs is None:
            layer_idxs = [3, 6, 9]
        self.layer_idxs = layer_idxs
        self.vision_dim = vision_dim
        self.ffn_dim = ffn_dim

        self.ia3_params = nn.ParameterDict({
            f'layer_{idx}': nn.ParameterDict({
                'l_k':   nn.Parameter(torch.ones(vision_dim)),
                'l_v':   nn.Parameter(torch.ones(vision_dim)),
                'l_ffn': nn.Parameter(torch.ones(ffn_dim)),
            })
            for idx in layer_idxs
        })

        print(f"  [MultiLayerIA3] vision_dim={vision_dim}, ffn_dim={ffn_dim}, layers={layer_idxs}")

    def get_params(self, layer_idx: int) -> dict:
        return self.ia3_params[f'layer_{layer_idx}']


class BiomedCLIP_IA3_Model(nn.Module):
    """IA3-enhanced BiomedCLIP: frozen backbone + channel-wise K/V/FFN scaling."""

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

        ia3_cfg = cfg.TRAINER.BIOMEDCLIP_IA3.IA3
        self.layer_idxs = ia3_cfg.LAYER_IDXS

        self.ia3_wrapper = MultiLayerIA3(vision_dim=768, ffn_dim=3072, layer_idxs=self.layer_idxs)

        print(f"[BIOMEDCLIP_IA3_Model] Visual embed_dim: {self.embed_dim}")
        print(f"[BIOMEDCLIP_IA3_Model] Target layers: {self.layer_idxs}")

    def get_text_embeddings(self, classnames=None):
        classnames = self.classnames
        texts = [f"a photo of {name}" for name in classnames]
        text_tokens = self.tokenizer(texts).to(next(self.parameters()).device)
        with torch.no_grad():
            text_embeds = self.text_encoder(text_tokens)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def _forward_with_ia3(self, image: torch.Tensor) -> torch.Tensor:
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
                ia3_params = self.ia3_wrapper.get_params(block_idx)
                x = vit_block_forward_with_ia3(block, x, ia3_params, drop_path_rate, device)
            else:
                x = block(x)

        x = self.visual.trunk.norm(x)
        cls_feature = x[:, 0]
        adapted_image_feature = self.visual.head(cls_feature)
        adapted_image_feature = F.normalize(adapted_image_feature, dim=-1)
        return adapted_image_feature

    def forward(self, image: torch.Tensor, label=None) -> dict:
        visual_features = self._forward_with_ia3(image)
        text_features = self.get_text_embeddings()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * visual_features @ text_features.t()

        output = {"logits": logits, "visual_features": visual_features}
        if label is not None:
            output["loss"] = F.cross_entropy(logits, label)
        return output
