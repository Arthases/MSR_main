import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class StaticAdapter(nn.Module):
    """x_out = x + residual_weight * bottleneck(x) with zero-init output."""

    def __init__(self, input_dim: int = 768, bottleneck_dim: int = 64, residual_weight: float = 0.1):
        super().__init__()
        self.residual_weight = residual_weight
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, input_dim),
        )
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_weight * self.adapter(x)


class MultiLayerStaticAdapter(nn.Module):
    """One StaticAdapter per target ViT block, inserted after attention projection."""

    def __init__(self, input_dim: int = 768, bottleneck_dim: int = 64,
                 layer_idxs: List[int] = None, residual_weight: float = 0.1):
        super().__init__()
        if layer_idxs is None:
            layer_idxs = [3, 6, 9]
        self.layer_idxs = layer_idxs

        self.adapters = nn.ModuleDict({
            f'layer_{idx}': StaticAdapter(input_dim, bottleneck_dim, residual_weight)
            for idx in layer_idxs
        })

        print(f"  [MultiLayerStaticAdapter] bottleneck_dim={bottleneck_dim}, "
              f"residual_weight={residual_weight}, layers={layer_idxs}")

    def forward(self, x: torch.Tensor, block_idx: int) -> torch.Tensor:
        layer_key = f'layer_{block_idx}'
        if layer_key in self.adapters:
            return self.adapters[layer_key](x)
        return x


def _drop_path(x, drop_prob, training):
    """Stochastic depth per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def _vit_block_with_adapter(block, x, adapter_wrapper, block_idx, drop_path_rate, device):
    """ViT Block with Adapter after attention and MLP sublayers."""
    x_residual = x
    x_norm = block.norm1(x)
    B, N, C = x_norm.shape

    qkv = block.attn.qkv(x_norm)
    qkv = qkv.reshape(B, N, 3, C).permute(2, 0, 1, 3)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * block.attn.scale
    attn = q @ k.transpose(-2, -1)
    attn = F.softmax(attn, dim=-1)
    attn = block.attn.attn_drop(attn)

    x_attn = attn @ v
    x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
    x_attn = block.attn.proj(x_attn)
    x_attn = adapter_wrapper(x_attn, block_idx)
    x_attn = block.attn.proj_drop(x_attn)
    x_attn = _drop_path(x_attn, drop_path_rate, block.training)
    x = x_residual + x_attn

    x_residual = x
    x_norm2 = block.norm2(x)
    x_mlp = block.mlp(x_norm2)
    x_mlp = adapter_wrapper(x_mlp, block_idx)
    x_mlp = _drop_path(x_mlp, drop_path_rate, block.training)
    x = x_residual + x_mlp
    return x


class BiomedCLIP_Adapter_Model(nn.Module):
    """Adapter-enhanced BiomedCLIP: frozen backbone + bottleneck adapters at target ViT blocks."""

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

        adapter_cfg = cfg.TRAINER.BIOMEDCLIP_ADAPTER.ADAPTER
        self.layer_idxs = adapter_cfg.LAYER_IDXS

        self.adapter_wrapper = MultiLayerStaticAdapter(
            input_dim=768, bottleneck_dim=adapter_cfg.BOTTLENECK_DIM,
            layer_idxs=self.layer_idxs, residual_weight=adapter_cfg.get('RESIDUAL_WEIGHT', 0.1),
        )

        print(f"[BiomedCLIP_Adapter_Model] Visual embed_dim: {self.embed_dim}")
        print(f"[BiomedCLIP_Adapter_Model] Target layers: {self.layer_idxs}")

    def get_text_embeddings(self, classnames=None):
        classnames = self.classnames
        texts = [f"a photo of {name}" for name in classnames]
        text_tokens = self.tokenizer(texts).to(next(self.parameters()).device)
        with torch.no_grad():
            text_embeds = self.text_encoder(text_tokens)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def _forward_with_adapter(self, image: torch.Tensor) -> torch.Tensor:
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
                x = _vit_block_with_adapter(block, x, self.adapter_wrapper, block_idx, drop_path_rate, device)
            else:
                x = block(x)

        x = self.visual.trunk.norm(x)
        cls_feature = x[:, 0]
        adapted_image_feature = self.visual.head(cls_feature)
        adapted_image_feature = F.normalize(adapted_image_feature, dim=-1)
        return adapted_image_feature

    def forward(self, image: torch.Tensor, label=None) -> dict:
        visual_features = self._forward_with_adapter(image)
        text_features = self.get_text_embeddings()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * visual_features @ text_features.t()

        output = {"logits": logits, "visual_features": visual_features}
        if label is not None:
            output["loss"] = F.cross_entropy(logits, label)
        return output
