import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


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


def _compute_lora_qkv(
    x: torch.Tensor,
    lora_params: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Compute LoRA QKV low-rank updates. Returns [B, N, 3D]."""
    B, N, D = x.shape
    rank = lora_params['q_A'].shape[0]

    p_drop = lora_params.get('dropout', 0.0)
    if isinstance(p_drop, torch.Tensor):
        p_drop = p_drop.item()
    if p_drop > 0.0:
        x = F.dropout(x, p=p_drop, training=x.requires_grad)

    q_down = F.linear(x, lora_params['q_A'])
    q_up   = F.linear(q_down, lora_params['q_B'])
    delta_q = q_up

    k_down = F.linear(x, lora_params['k_A'])
    k_up   = F.linear(k_down, lora_params['k_B'])
    delta_k = k_up

    v_down = F.linear(x, lora_params['v_A'])
    v_up   = F.linear(v_down, lora_params['v_B'])
    delta_v = v_up

    delta_qkv = torch.cat([delta_q, delta_k, delta_v], dim=2)
    return delta_qkv


def vit_block_forward_with_lora(
    block,
    x: torch.Tensor,
    lora_params: Dict[str, torch.Tensor],
    drop_path_rate: float = 0.0,
    device: torch.device = None,
) -> torch.Tensor:
    """ViT Block forward with LoRA QKV injection: qkv_out = qkv_frozen + alpha * B @ A @ x."""
    if device is None:
        device = x.device

    x_residual = x

    x_norm = block.norm1(x)
    B, N, C = x_norm.shape

    qkv_0 = block.attn.qkv(x_norm)
    qkv_lora = _compute_lora_qkv(x_norm, lora_params, device)

    alpha = lora_params.get('alpha', 1.0)
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    qkv = qkv_0 + alpha * qkv_lora

    qkv = qkv.reshape(B, N, 3, C).permute(2, 0, 1, 3)
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * block.attn.scale
    attn = q @ k.transpose(-2, -1)
    attn = F.softmax(attn, dim=-1)
    attn = block.attn.attn_drop(attn)

    x_attn = attn @ v
    x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
    x_attn = block.attn.proj(x_attn)
    x_attn = block.attn.proj_drop(x_attn)

    x_attn = _drop_path(x_attn, drop_path_rate, block.training)
    x = x_residual + x_attn

    x_residual = x
    x_norm2 = block.norm2(x)
    x_mlp = block.mlp(x_norm2)
    x_mlp = _drop_path(x_mlp, drop_path_rate, block.training)
    x = x_residual + x_mlp

    return x



def vit_block_forward_with_ia3(
    block,
    x: torch.Tensor,
    ia3_params: Dict[str, torch.Tensor],
    drop_path_rate: float = 0.0,
    device: torch.device = None,
) -> torch.Tensor:
    """ViT Block forward with IA3 scaling: k = l_k*k, v = l_v*v, ffn = l_ffn*GELU(fc1(x))."""
    if device is None:
        device = x.device

    x_residual = x

    x_norm = block.norm1(x)
    B, N, C = x_norm.shape

    qkv = block.attn.qkv(x_norm)
    qkv = qkv.reshape(B, N, 3, C).permute(2, 0, 1, 3)
    q, k, v = qkv[0], qkv[1], qkv[2]

    l_k = ia3_params['l_k'].to(device)
    l_v = ia3_params['l_v'].to(device)

    k = l_k * k
    v = l_v * v

    q = q * block.attn.scale
    attn = q @ k.transpose(-2, -1)
    attn = F.softmax(attn, dim=-1)
    attn = block.attn.attn_drop(attn)

    x_attn = attn @ v
    x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
    x_attn = block.attn.proj(x_attn)
    x_attn = block.attn.proj_drop(x_attn)

    x_attn = _drop_path(x_attn, drop_path_rate, block.training)
    x = x_residual + x_attn

    x_residual = x
    x_norm2 = block.norm2(x)

    l_ffn = ia3_params['l_ffn'].to(device)
    l_ffn_scaled = l_ffn.unsqueeze(0).unsqueeze(0)

    mid = block.mlp.fc1(x_norm2)
    act = block.mlp.act if hasattr(block.mlp, 'act') else F.gelu
    mid = act(mid)
    mid = l_ffn_scaled * mid
    if hasattr(block.mlp, 'drop1'):
        mid = block.mlp.drop1(mid)
    elif hasattr(block.mlp, 'drop'):
        mid = block.mlp.drop(mid)
    x_mlp = block.mlp.fc2(mid)
    if hasattr(block.mlp, 'drop2'):
        x_mlp = block.mlp.drop2(x_mlp)
    elif hasattr(block.mlp, 'drop') and not hasattr(block.mlp, 'drop1'):
        x_mlp = block.mlp.drop(x_mlp)

    x_mlp = _drop_path(x_mlp, drop_path_rate, block.training)
    x = x_residual + x_mlp

    return x
