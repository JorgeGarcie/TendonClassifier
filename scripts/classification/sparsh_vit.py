# Vendored Sparsh ViT from facebookresearch/sparsh (Apache 2.0)
# https://github.com/facebookresearch/sparsh
#
# Self-contained: no xformers dependency, no external tactile_ssl imports.
# Only needs: torch, einops, numpy.

import logging
import math
from functools import partial
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import trunc_normal_

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities (from tactile_ssl.utils)
# ---------------------------------------------------------------------------

def create_ndgrid(
    resolution: List[int],
    device: torch.device = torch.device("cpu"),
    normalized_coords: bool = True,
    indexing: Literal["xy", "ij"] = "ij",
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    assert len(resolution) <= 3, "Only up to 3D grids are supported"
    axes = []
    if normalized_coords:
        for res in resolution:
            axes.append(torch.linspace(-1, 1, res + 1, dtype=dtype, device=device)[:-1])
    else:
        for res in resolution:
            axes.append(torch.arange(0, res, dtype=dtype, device=device))
    grid = torch.stack(torch.meshgrid(*axes, indexing=indexing), dim=-1)
    if len(resolution) == 2:
        grid = einops.rearrange(grid, "y x ... -> (y x) ...")
    elif len(resolution) == 3:
        grid = einops.rearrange(grid, "z y x ... -> (z y x) ...")
    return grid


def apply_masks(x, masks, concat=True):
    all_x = []
    for mask in masks:
        mask_keep = einops.repeat(mask, "b n -> b n d", d=x.size(-1))
        all_x.append(torch.gather(x, dim=-2, index=mask_keep))
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


# ---------------------------------------------------------------------------
# DropPath
# ---------------------------------------------------------------------------

def _drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# LayerScale
# ---------------------------------------------------------------------------

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-5,
                 inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# ---------------------------------------------------------------------------
# Mlp
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: Callable[..., nn.Module] = nn.GELU,
                 drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# SwiGLU FFN (fallback without xformers)
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: Callable[..., nn.Module] = None,
                 drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class SwiGLUFFNFused(SwiGLUFFN):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: Callable[..., nn.Module] = None,
                 drop: float = 0.0, bias: bool = True) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(in_features=in_features, hidden_features=hidden_features,
                         out_features=out_features, bias=bias)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 proj_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    """Falls back to vanilla Attention when xformers is unavailable."""
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if attn_bias is not None:
            raise AssertionError("xFormers is required for attn_bias (nested tensors)")
        return super().forward(x)


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

def _drop_add_residual_stochastic_depth(x, residual_func, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]
    residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, proj_bias: bool = True,
                 ffn_bias: bool = True, drop: float = 0.0,
                 attn_drop: float = 0.0, init_values=None,
                 drop_path: float = 0.0,
                 act_layer: Callable[..., nn.Module] = nn.GELU,
                 norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
                 attn_class: Callable[..., nn.Module] = Attention,
                 ffn_layer: Callable[..., nn.Module] = Mlp) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x):
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            x = _drop_add_residual_stochastic_depth(
                x, residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio)
            x = _drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


class NestedTensorBlock(Block):
    """Without xformers, only supports single-tensor forward (falls back to Block)."""
    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        raise AssertionError("xFormers is required for nested tensor (list) inputs")


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------

def _make_2tuple(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 2
        return x
    return (x, x)


def _make_tuple(x):
    if isinstance(x, (tuple, list)):
        return x
    return (x,)


class PatchEmbed(nn.Module):
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 in_chans: int = 3, embed_dim: int = 768,
                 norm_layer: Optional[Callable] = None,
                 flatten_embedding: bool = True) -> None:
        super().__init__()
        image_HW = _make_2tuple(img_size)
        patch_HW = _make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])
        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches: int = int(patch_grid_size[0] * patch_grid_size[1])
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x


# ---------------------------------------------------------------------------
# Sinusoidal Positional Embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbed(nn.Module):
    def __init__(self, size: Union[int, List[int]],
                 stride: Union[int, List[int]],
                 embed_dim: int = 768, logspace: bool = False) -> None:
        super().__init__()
        size = _make_tuple(size)
        stride = _make_tuple(stride)
        assert len(size) < 4
        assert len(size) == len(stride)
        patch_grid_size = [s // stride[i] for i, s in enumerate(size)]
        self.embed_dim = embed_dim
        self.patches_resolution = patch_grid_size
        self.num_patches = int(np.prod(patch_grid_size))
        self.num_bands = math.ceil(self.embed_dim / (2 * len(size)))
        resolution = 10000
        if logspace:
            frequency_bands = torch.stack([
                torch.logspace(0.0, -math.log2(resolution / 2.0),
                               self.num_bands + 1, base=2)[:-1]
                for _ in range(len(size))
            ], dim=0)
        else:
            frequency_bands = torch.stack([
                torch.linspace(0, 1.0, steps=self.num_bands + 1)[:-1]
                for _ in range(len(size))
            ], dim=0)
            frequency_bands = resolution ** -frequency_bands
        self.register_buffer("frequency_bands", frequency_bands)
        self.register_buffer("cached_encoding", None, persistent=False)

    def forward(self, device: torch.device, normalized_coords: bool = False):
        if self.cached_encoding is not None:
            if self.cached_encoding.device == device:
                return self.cached_encoding
            else:
                return self.cached_encoding.to(device, non_blocking=True)
        grid = create_ndgrid(self.patches_resolution, device=device,
                             normalized_coords=normalized_coords)
        if grid.dim() < 2:
            grid = grid[..., None]
        freq_bands_buf = self.get_buffer("frequency_bands")
        features = grid[..., None] * freq_bands_buf
        encoded_pos = torch.cat([torch.sin(features), torch.cos(features)], dim=-1)
        encoded_pos = encoded_pos.flatten(-2, -1)
        self.cached_encoding = encoded_pos[..., :self.embed_dim]
        return self.cached_encoding


# ---------------------------------------------------------------------------
# VisionTransformer
# ---------------------------------------------------------------------------

def named_apply(fn, module, name="", depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name,
                    depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = (224, 224),
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: int = 1,
        embed_layer=PatchEmbed,
        pos_embed_fn: Literal["sinusoidal", "learned"] = "learned",
        act_layer=nn.GELU,
        block_fn=None,
        ffn_layer="mlp",
        block_chunks: int = 0,
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
    ):
        super().__init__()
        if block_fn is None:
            block_fn = Block
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.img_size = img_size
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.pos_embed_fn = pos_embed_fn

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens else None
        )
        if pos_embed_fn == "sinusoidal":
            self.pos_embed = SinusoidalEmbed(
                list(self.img_size), [self.patch_size, self.patch_size],
                embed_dim=self.embed_dim)
        elif pos_embed_fn == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer in ("swiglufused", "swiglu"):
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            ffn_layer = lambda *args, **kwargs: nn.Identity()
        else:
            raise NotImplementedError(f"Unknown ffn_layer: {ffn_layer}")

        blocks_list = [
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                     drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                     ffn_layer=ffn_layer, init_values=init_values)
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i:i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.init_weights()
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self):
        if self.pos_embed_fn == "learned":
            trunc_normal_(self.pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, img_shape, img_dtype, device):
        if self.pos_embed_fn == "sinusoidal":
            pos_embed = self.pos_embed(device).float().unsqueeze(0)
        elif self.pos_embed_fn == "learned":
            pos_embed = self.pos_embed.float()
        else:
            raise NotImplementedError

        _, _, h, w = img_shape
        if h == self.img_size[0] and w == self.img_size[1]:
            return pos_embed

        dim = pos_embed.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, w0, h0, dim).permute(0, 3, 1, 2),
            mode="bicubic", antialias=self.interpolate_antialias, size=(w0, h0))
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed.to(img_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        pos_encoding = self.interpolate_pos_encoding(x.shape, x.dtype, device=x.device)
        x = self.patch_embed(x)
        x = x + pos_encoding
        if masks is not None:
            x = apply_masks(x, masks)
        if self.register_tokens is not None:
            x = torch.cat((self.register_tokens.expand(x.shape[0], -1, -1), x), dim=1)
        return x

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {
            "x_norm_regtokens": x_norm[:, :self.num_register_tokens],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens:],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret["x_norm_patchtokens"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    return VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens, **kwargs)
