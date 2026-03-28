"""DINOv3 ViT-H/16+ — Pure PyTorch backbone and tagger head.

All hyperparameters match facebook/dinov3-vith16plus-pretrain-lvd1689m.
State-dict keys are intentionally identical to the HuggingFace transformers
layout so .safetensors checkpoints load without remapping.

No external ML framework dependency (no timm, no transformers).
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Architecture constants
# =============================================================================

D_MODEL     = 1280
N_HEADS     = 20
HEAD_DIM    = D_MODEL // N_HEADS   # 64
N_LAYERS    = 32
D_FFN       = 5120
N_REGISTERS = 4
PATCH_SIZE  = 16
ROPE_THETA  = 100.0
ROPE_RESCALE = 2.0
LN_EPS      = 1e-5
LAYERSCALE  = 1.0


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _patch_coords_cached(h: int, w: int, device_str: str) -> torch.Tensor:
    """Normalised [-1,+1] patch-centre coordinates (float32, cached)."""
    device = torch.device(device_str)
    cy = torch.arange(0.5, h, dtype=torch.float32, device=device) / h
    cx = torch.arange(0.5, w, dtype=torch.float32, device=device) / w
    coords = torch.stack(torch.meshgrid(cy, cx, indexing="ij"), dim=-1).flatten(0, 1)
    coords = 2.0 * coords - 1.0   # [0,1] -> [-1,+1]
    coords = coords * ROPE_RESCALE
    return coords  # [h*w, 2]


def _build_rope(h_patches: int, w_patches: int,
                dtype: torch.dtype, device: torch.device):
    """Return (cos, sin) of shape [1, 1, h*w, HEAD_DIM] for broadcasting."""
    coords = _patch_coords_cached(h_patches, w_patches, str(device))  # [P, 2]
    inv_freq = 1.0 / (ROPE_THETA ** torch.arange(
        0, 1, 4 / HEAD_DIM, dtype=torch.float32, device=device))      # [D/4]
    angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]  # [P, 2, D/4]
    angles = angles.flatten(1, 2).tile(2)                                 # [P, D]
    cos = torch.cos(angles).to(dtype).unsqueeze(0).unsqueeze(0)  # [1,1,P,D]
    sin = torch.sin(angles).to(dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE only to patch tokens (skip CLS + register prefix)."""
    n_pre = 1 + N_REGISTERS
    q_pre, q_pat = q[..., :n_pre, :], q[..., n_pre:, :]
    k_pre, k_pat = k[..., :n_pre, :], k[..., n_pre:, :]
    q_pat = q_pat * cos + _rotate_half(q_pat) * sin
    k_pat = k_pat * cos + _rotate_half(k_pat) * sin
    return torch.cat([q_pre, q_pat], dim=-2), torch.cat([k_pre, k_pat], dim=-2)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM ** -0.5)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, D_MODEL))


class _GatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(D_MODEL, D_FFN, bias=True)
        self.up_proj   = nn.Linear(D_MODEL, D_FFN, bias=True)
        self.down_proj = nn.Linear(D_FFN,   D_MODEL, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1        = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.attention    = _Attention()
        self.layer_scale1 = nn.Parameter(torch.full((D_MODEL,), LAYERSCALE))
        self.norm2        = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.mlp          = _GatedMLP()
        self.layer_scale2 = nn.Parameter(torch.full((D_MODEL,), LAYERSCALE))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), cos, sin) * self.layer_scale1
        x = x + self.mlp(self.norm2(x)) * self.layer_scale2
        return x


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class _Embeddings(nn.Module):
    """Patch + CLS + register token embeddings.
    Key names match HF: embeddings.cls_token, embeddings.register_tokens,
    embeddings.patch_embeddings.{weight,bias}.
    """

    def __init__(self):
        super().__init__()
        self.cls_token       = nn.Parameter(torch.empty(1, 1, D_MODEL))
        self.mask_token      = nn.Parameter(torch.zeros(1, 1, D_MODEL))  # unused at inference
        self.register_tokens = nn.Parameter(torch.empty(1, N_REGISTERS, D_MODEL))
        self.patch_embeddings = nn.Conv2d(3, D_MODEL, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.shape[0]
        dtype = self.patch_embeddings.weight.dtype
        patches = self.patch_embeddings(pixel_values.to(dtype)).flatten(2).transpose(1, 2)
        cls  = self.cls_token.expand(B, -1, -1)
        regs = self.register_tokens.expand(B, -1, -1)
        return torch.cat([cls, regs, patches], dim=1)


# ---------------------------------------------------------------------------
# Full backbone
# ---------------------------------------------------------------------------

class DINOv3ViTH(nn.Module):
    """DINOv3 ViT-H/16+ backbone.

    Accepts any H, W that are multiples of 16.
    Returns last_hidden_state [B, 1+R+P, D_MODEL].
    Token layout: [CLS, reg_0..reg_3, patch_0..patch_N].
    """

    def __init__(self):
        super().__init__()
        self.embeddings = _Embeddings()
        self.layer = nn.ModuleList([_Block() for _ in range(N_LAYERS)])
        self.norm  = nn.LayerNorm(D_MODEL, eps=LN_EPS)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                               strict, missing_keys, unexpected_keys, error_msgs):
        # HF stores layer_scale as a sub-module with a "lambda1" parameter;
        # we store it as a plain Parameter directly on _Block.
        # Remap "layer.i.layer_scale{1,2}.lambda1" -> "layer.i.layer_scale{1,2}"
        for k in list(state_dict.keys()):
            if k.startswith(prefix) and ".layer_scale" in k and k.endswith(".lambda1"):
                new_k = k[:-len(".lambda1")]
                state_dict[new_k] = state_dict.pop(k)
        # Drop rope_embeddings buffer (computed on-the-fly)
        for k in list(state_dict.keys()):
            if k.startswith(prefix) and "rope_embeddings" in k:
                state_dict.pop(k)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, _, H, W = pixel_values.shape
        x = self.embeddings(pixel_values)  # [B, 1+R+P, D]

        h_p, w_p = H // PATCH_SIZE, W // PATCH_SIZE
        cos, sin = _build_rope(h_p, w_p, x.dtype, pixel_values.device)

        for block in self.layer:
            x = block(x, cos, sin)

        return self.norm(x)


# =============================================================================
# Tagger head
# =============================================================================

class DINOv3TaggerModel(nn.Module):
    """DINOv3 ViT-H/16+ backbone + linear projection head.

    features = concat(CLS, reg_0..reg_3) -> [B, (1+R)*D]
    projection: Linear -> [B, num_tags]
    """

    def __init__(self, num_tags: int, projection_bias: bool = False):
        super().__init__()
        self.backbone   = DINOv3ViTH()
        self.projection = nn.Linear((1 + N_REGISTERS) * D_MODEL, num_tags, bias=projection_bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden   = self.backbone(pixel_values)                      # [B, S, D]
        cls      = hidden[:, 0, :]                                  # [B, D]
        regs     = hidden[:, 1: 1 + N_REGISTERS, :].flatten(1)     # [B, R*D]
        features = torch.cat([cls, regs], dim=-1)                   # [B, (1+R)*D]
        return self.projection(features.float())                     # fp32 for stability
