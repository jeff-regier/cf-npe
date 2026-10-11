"""2D ResNets for the strong-lensing networks.

``ResNet2DDown`` downsamples (image encoder); ``ResNet2DSame`` keeps resolution (Stage-1 mean
head and NSF coupling conditioner on the source grid).
"""

from __future__ import annotations

import math
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cf_npe.networks.posterior import _gn

_ACTIVATIONS = {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}

PaddingMode = Literal["zeros", "reflect", "replicate", "circular"]


def _act(kind: str, x: Tensor) -> Tensor:
    return _ACTIVATIONS[kind](x)


def _drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask = mask.floor_()
    return x.div(keep) * mask


def _block_drop_path_rates(n_blocks: int, drop_path_max: float) -> list[float]:
    """Linear schedule from 0 to drop_path_max (inclusive) across n_blocks."""
    denom = max(n_blocks - 1, 1)
    return [drop_path_max * i / denom for i in range(n_blocks)]


def _evenly_spaced_positions(n_blocks: int, n_attention: int) -> list[int]:
    """0-indexed block indices, evenly spaced, after which attention layers are applied.

    Raises ValueError when rounding cannot place n_attention distinct positions.
    """
    if n_attention <= 0:
        return []
    if n_attention > n_blocks:
        raise ValueError(f"n_attention={n_attention} exceeds n_blocks={n_blocks}")
    positions = [int(round((i + 1) * n_blocks / (n_attention + 1))) - 1 for i in range(n_attention)]
    if positions[0] < 0 or len(set(positions)) != len(positions):
        raise ValueError(
            f"duplicate attention positions {positions} for "
            f"n_blocks={n_blocks}, n_attention={n_attention}"
        )
    return positions


def sin_cos_pos_2d(n_side: int) -> Tensor:
    """(1, 4, n, n) map of (sin x, cos x, sin y, cos y), coordinates evenly spaced on [-pi, pi).

    Gives a translation-equivariant conv absolute pixel coordinates, e.g. for a centered lens.
    """
    grid = torch.arange(n_side) * (2 * math.pi / n_side) - math.pi
    ys, xs = torch.meshgrid(grid, grid, indexing="ij")
    channels = [torch.sin(xs), torch.cos(xs), torch.sin(ys), torch.cos(ys)]
    return torch.stack(channels, dim=0).unsqueeze(0)


class _SEBlock(nn.Module):
    """Squeeze-and-excite channel attention. ``fc2`` is zero-initialized, so at
    start it scales every channel by sigmoid(4) ≈ 0.98, close to the identity."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s) + 4.0)
        return x * s.unsqueeze(-1).unsqueeze(-1)


class _ResBlock2D(nn.Module):
    """Pre-activation residual block with a 1×1 projection shortcut when shape changes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout: float,
        activation: str,
        norm_groups: int,
        padding_mode: PaddingMode,
        use_se: bool,
        drop_path_rate: float,
    ):
        super().__init__()
        self.norm1 = _gn(in_channels, norm_groups)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, padding_mode=padding_mode
        )
        self.norm2 = _gn(out_channels, norm_groups)
        self.drop = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode=padding_mode)
        need_proj = stride != 1 or in_channels != out_channels
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride) if need_proj else nn.Identity()
        )
        self.se = _SEBlock(out_channels) if use_se else nn.Identity()
        self.activation = activation
        self.drop_path_rate = drop_path_rate

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(_act(self.activation, self.norm1(x)))
        h = self.norm2(h)
        h = self.se(self.conv2(self.drop(_act(self.activation, h))))
        h = _drop_path(h, self.drop_path_rate, self.training)
        return h + self.shortcut(x)


class ResNet2DDown(nn.Module):
    """Pure downsampling 2D ResNet. Input H×W → output (H/2^n_downsamples)×(same)."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        n_downsamples: int,
        blocks_per_level: int,
        dropout: float,
        activation: str,
        norm_groups: int,
        padding_mode: PaddingMode,
        use_se: bool,
        drop_path_max: float,
    ):
        super().__init__()
        c = [base_channels * (2**i) for i in range(n_downsamples + 1)]
        self.stem = nn.Conv2d(in_channels, c[0], 3, padding=1, padding_mode=padding_mode)
        n_total = (n_downsamples + 1) * blocks_per_level
        dp_rates = _block_drop_path_rates(n_total, drop_path_max)
        levels: list[nn.Module] = []
        idx = 0
        for i in range(n_downsamples + 1):
            blocks: list[nn.Module] = []
            for k in range(blocks_per_level):
                downsample = i > 0 and k == 0
                blocks.append(
                    _ResBlock2D(
                        c[i - 1] if downsample else c[i],
                        c[i],
                        stride=2 if downsample else 1,
                        dropout=dropout,
                        activation=activation,
                        norm_groups=norm_groups,
                        padding_mode=padding_mode,
                        use_se=use_se,
                        drop_path_rate=dp_rates[idx],
                    )
                )
                idx += 1
            levels.append(nn.ModuleList(blocks))
        self.levels = nn.ModuleList(levels)
        self.out_channels = c[-1]

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem(x)
        for level in self.levels:
            level = cast(nn.ModuleList, level)
            for blk in level:
                h = blk(h)
        return h


class _SelfAttention2D(nn.Module):
    """Residual self-attention over spatial positions.

    Sinusoidal 2D position embeddings are added to Q and K only, so V carries no positional bias.
    """

    pos_emb: Tensor

    def __init__(self, channels: int, n_heads: int, n_side: int, norm_groups: int):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.norm = _gn(channels, norm_groups)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        pos = self._sinusoidal_2d_pos(n_side, channels)
        self.register_buffer("pos_emb", pos)

    @staticmethod
    def _sinusoidal_2d_pos(n: int, dim: int) -> Tensor:
        half = dim // 2
        ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        ys = ys.flatten().float()
        xs = xs.flatten().float()
        d_q = half // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(d_q).float() / d_q)
        pe = torch.zeros(n * n, dim)
        pe[:, 0:d_q] = torch.sin(xs.unsqueeze(-1) * freqs)
        pe[:, d_q : 2 * d_q] = torch.cos(xs.unsqueeze(-1) * freqs)
        pe[:, half : half + d_q] = torch.sin(ys.unsqueeze(-1) * freqs)
        pe[:, half + d_q : half + 2 * d_q] = torch.cos(ys.unsqueeze(-1) * freqs)
        return pe  # (n*n, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)  # (B, HW, C)
        h_pos = h + self.pos_emb.unsqueeze(0)
        w, bias = self.qkv.weight, self.qkv.bias
        q, k = F.linear(h_pos, w[: 2 * C], bias[: 2 * C]).chunk(2, dim=-1)
        v = F.linear(h, w[2 * C :], bias[2 * C :])
        q = q.view(B, H * W, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, H * W, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, H * W, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)
        return out.transpose(1, 2).view(B, C, H, W) + x


class ResNet2DSame(nn.Module):
    """Fixed-resolution 2D ResNet; the 1×1 readout is zero-initialized, so it outputs zeros at init.

    ``use_pos_encoding`` appends a sin/cos coordinate map to the input; ``use_global_pool``
    appends the input's spatial mean broadcast back to the grid.
    """

    _pos_enc: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        n_blocks: int,
        dropout: float,
        n_attention: int,
        attention_heads: int,
        n_side: int,
        activation: str,
        norm_groups: int,
        padding_mode: PaddingMode,
        use_se: bool,
        drop_path_max: float,
        use_pos_encoding: bool,
        use_global_pool: bool,
    ):
        super().__init__()
        self.use_pos_encoding = use_pos_encoding
        self.use_global_pool = use_global_pool
        if use_pos_encoding:
            self.register_buffer("_pos_enc", sin_cos_pos_2d(n_side))
        stem_in = (
            in_channels + (4 if use_pos_encoding else 0) + (in_channels if use_global_pool else 0)
        )
        self.stem = nn.Conv2d(stem_in, width, 3, padding=1, padding_mode=padding_mode)
        dp_rates = _block_drop_path_rates(n_blocks, drop_path_max)
        self.blocks = nn.ModuleList(
            [
                _ResBlock2D(
                    width,
                    width,
                    stride=1,
                    dropout=dropout,
                    activation=activation,
                    norm_groups=norm_groups,
                    padding_mode=padding_mode,
                    use_se=use_se,
                    drop_path_rate=dp_rates[i],
                )
                for i in range(n_blocks)
            ]
        )
        positions = _evenly_spaced_positions(n_blocks, n_attention)
        self.attentions = nn.ModuleDict(
            {
                str(p): _SelfAttention2D(
                    width, n_heads=attention_heads, n_side=n_side, norm_groups=norm_groups
                )
                for p in positions
            }
        )
        self.attention_position_set = set(positions)
        self.readout = nn.Conv2d(width, out_channels, kernel_size=1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(cast(Tensor, self.readout.bias))

    def forward(self, x: Tensor) -> Tensor:
        if self.use_pos_encoding or self.use_global_pool:
            parts = [x]
            if self.use_pos_encoding:
                parts.append(self._pos_enc.expand(x.shape[0], -1, -1, -1))
            if self.use_global_pool:
                parts.append(x.mean(dim=(2, 3), keepdim=True).expand_as(x))
            x = torch.cat(parts, dim=1)
        h = self.stem(x)
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if i in self.attention_position_set:
                h = self.attentions[str(i)](h)
        return self.readout(h)
