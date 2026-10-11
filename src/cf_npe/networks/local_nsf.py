"""Coupling normalizing flow over a latent laid out on a square grid, with ResNet2D conditioners."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from cf_npe.networks.posterior import (
    RQSTransform,
    SamplingMomentMixin,
    _spline_params_per_dim,
)
from cf_npe.networks.resnet2d import ResNet2DSame

# Samples per coupling-stack pass in JointNSF.rsample; bounds activation memory of
# the spatial conditioners, whose per-pass tensor is (batch * chunk, C, n, n).
_SAMPLES_CHUNK = 16


def _build_level_indices(n_side: int, even_fixed: bool, mask_kind: str) -> tuple[Tensor, Tensor]:
    """(fixed_idx, transform_idx) for one coupling: alternate grid rows (``"stripe_row"``) or
    columns (``"stripe_col"``) are fixed."""
    levels = torch.arange(n_side * n_side)
    stripe = {"stripe_row": levels // n_side, "stripe_col": levels % n_side}[mask_kind]
    fixed = (stripe % 2 == 0) == even_fixed
    return levels[fixed], levels[~fixed]


@dataclass
class ResNet2dConditionerConfig:
    """Capacity of the ``ResNet2DSame`` coupling conditioner."""

    n_blocks: int
    n_attention: int
    attention_heads: int
    dropout: float
    use_se: bool
    drop_path_max: float
    use_pos_encoding: bool
    use_global_pool: bool


class ResNet2dConditioner(nn.Module):
    """ResNet2D coupling conditioner on the ``n_side × n_side`` latent grid.

    Input channels: the fixed half of ``z`` on the full grid (zeros elsewhere), the spatial
    context, and a binary mask. The backbone's zero-init readout makes each coupling start at
    the identity.
    """

    fixed_level_idx: Tensor
    transform_level_idx: Tensor
    fixed_mask: Tensor

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        n_params_per_dim: int,
        fixed_idx: Tensor,
        transform_idx: Tensor,
        n_side: int,
        cfg: ResNet2dConditionerConfig,
    ):
        super().__init__()
        self.dim_z = n_side * n_side
        self.n_side = n_side
        self.n_params_per_dim = n_params_per_dim
        self.register_buffer("fixed_level_idx", fixed_idx)
        self.register_buffer("transform_level_idx", transform_idx)
        mask = torch.zeros(self.dim_z)
        mask[fixed_idx] = 1.0
        self.register_buffer("fixed_mask", mask.view(1, 1, n_side, n_side))
        self.net = ResNet2DSame(
            in_channels=context_dim + 2,
            out_channels=n_params_per_dim,
            width=hidden_dim,
            n_blocks=cfg.n_blocks,
            dropout=cfg.dropout,
            n_attention=cfg.n_attention,
            attention_heads=cfg.attention_heads,
            n_side=n_side,
            activation="gelu",
            norm_groups=32,
            padding_mode="zeros",
            use_se=cfg.use_se,
            drop_path_max=cfg.drop_path_max,
            use_pos_encoding=cfg.use_pos_encoding,
            use_global_pool=cfg.use_global_pool,
        )

    def forward(self, z_fixed: Tensor, context: Tensor) -> Tensor:
        leading = z_fixed.shape[:-1]
        n = self.n_side
        k_fixed = len(self.fixed_level_idx)
        k_out = len(self.transform_level_idx)
        b_flat = z_fixed.reshape(-1, k_fixed)
        b = b_flat.shape[0]
        ctx = context.reshape(-1, context.shape[-3], n, n)

        # Out-of-place scatter: vmap rejects in-place __setitem__ with an index tensor shared
        # across the vmapped dim.
        z_full = torch.zeros(b, 1, self.dim_z, device=z_fixed.device, dtype=z_fixed.dtype)
        z_full = z_full.index_copy(2, self.fixed_level_idx, b_flat.unsqueeze(1))
        mask = self.fixed_mask.expand(b, -1, -1, -1)
        inp = torch.cat([z_full.view(b, 1, n, n), ctx, mask], dim=1)
        out = self.net(inp).view(b, self.n_params_per_dim, self.dim_z)
        out = out[:, :, self.transform_level_idx].transpose(1, 2)
        return out.reshape(*leading, k_out, self.n_params_per_dim)


class CouplingLayer(nn.Module):
    """One spline coupling transform over a stripe split of the grid."""

    fixed_idx: Tensor
    transform_idx: Tensor

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        n_bins: int,
        even_fixed: bool,
        n_side: int,
        resnet_cfg: ResNet2dConditionerConfig,
        tail_bound: float,
        mask_kind: str,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.tail_bound = tail_bound
        fixed_idx, transform_idx = _build_level_indices(n_side, even_fixed, mask_kind)
        self.register_buffer("fixed_idx", fixed_idx)
        self.register_buffer("transform_idx", transform_idx)
        self.conditioner = ResNet2dConditioner(
            context_dim,
            hidden_dim,
            _spline_params_per_dim(n_bins),
            fixed_idx,
            transform_idx,
            n_side,
            resnet_cfg,
        )

    def _get_spline(
        self,
        z_fixed: Tensor,
        context: Tensor,
    ) -> RQSTransform:
        params = self.conditioner(z_fixed, context)
        w, h, d = params.split(
            [self.n_bins, self.n_bins, self.n_bins - 1],
            dim=-1,
        )
        return RQSTransform(w, h, d, bound=self.tail_bound)

    def forward_and_ladj(
        self,
        z: Tensor,
        context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """z: (*batch, dim_z), context: (*batch, context_dim, n_side, n_side) -> (z_out, ladj)."""
        z_fixed = z[..., self.fixed_idx]
        z_transform = z[..., self.transform_idx]
        spline = self._get_spline(z_fixed, context)
        z_new, ladj = spline.call_and_ladj(z_transform)
        z_out = torch.empty_like(z)
        z_out[..., self.fixed_idx] = z_fixed
        z_out[..., self.transform_idx] = z_new
        return z_out, ladj.sum(-1)

    def inverse(
        self,
        z: Tensor,
        context: Tensor,
    ) -> Tensor:
        z_fixed = z[..., self.fixed_idx]
        z_transform = z[..., self.transform_idx]
        spline = self._get_spline(z_fixed, context)
        z_orig = spline._inverse(z_transform)
        z_out = torch.empty_like(z)
        z_out[..., self.fixed_idx] = z_fixed
        z_out[..., self.transform_idx] = z_orig
        return z_out


class JointNSF(nn.Module):
    """Stack of coupling layers with standard normal base over the n_side² grid cells."""

    def __init__(
        self,
        context_dim: int,
        n_side: int,
        resnet_cfg: ResNet2dConditionerConfig,
        n_transforms: int,
        n_bins: int,
        hidden_dim: int,
        tail_bound: float,
        mask_kind: str,
    ):
        super().__init__()
        self.dim_z = n_side * n_side
        self._log_norm = 0.5 * self.dim_z * math.log(2 * math.pi)
        self.layers = nn.ModuleList(
            [
                CouplingLayer(
                    context_dim=context_dim,
                    hidden_dim=hidden_dim,
                    n_bins=n_bins,
                    even_fixed=(i % 2 == 0),
                    n_side=n_side,
                    tail_bound=tail_bound,
                    mask_kind=mask_kind,
                    resnet_cfg=resnet_cfg,
                )
                for i in range(n_transforms)
            ]
        )

    def log_prob(self, z: Tensor, context: Tensor) -> Tensor:
        total_ladj = torch.zeros(
            z.shape[:-1],
            device=z.device,
        )
        for layer in self.layers:
            coupling = cast(CouplingLayer, layer)
            z, ladj = coupling.forward_and_ladj(z, context)
            total_ladj = total_ladj + ladj
        base_lp = -0.5 * z.pow(2).sum(-1) - self._log_norm
        return base_lp + total_ladj

    def rsample(self, context: Tensor, n_samples: int) -> Tensor:
        """(batch, n_samples, dim_z) samples, drawn ``_SAMPLES_CHUNK`` at a time."""
        batch = context.shape[0]
        n_trailing = context.dim() - 1
        chunks: list[Tensor] = []
        for start in range(0, n_samples, _SAMPLES_CHUNK):
            k = min(_SAMPLES_CHUNK, n_samples - start)
            z = torch.randn(batch, k, self.dim_z, device=context.device)
            ctx = context.unsqueeze(1).expand(-1, k, *(-1,) * n_trailing)
            for layer in reversed(self.layers):
                coupling = cast(CouplingLayer, layer)
                z = coupling.inverse(z, ctx)
            chunks.append(z)
        return torch.cat(chunks, dim=1)


class LocalNSFPosterior(SamplingMomentMixin, nn.Module):
    """Joint NSF posterior on a 2D latent grid with ResNet2D coupling conditioners."""

    def __init__(
        self,
        encoder: nn.Module,
        head_dim: int,
        hidden_dim: int,
        n_side: int,
        resnet_cfg: ResNet2dConditionerConfig,
        n_transforms: int,
        n_bins: int,
        tail_bound: float,
        mask_kind: str,
    ):
        super().__init__()
        self.encoder = encoder
        self.flow = JointNSF(
            context_dim=head_dim,
            n_transforms=n_transforms,
            n_bins=n_bins,
            hidden_dim=hidden_dim,
            n_side=n_side,
            tail_bound=tail_bound,
            mask_kind=mask_kind,
            resnet_cfg=resnet_cfg,
        )

    def log_prob(self, x: Tensor, z: Tensor) -> Tensor:
        h = self.encoder(x)
        return self.flow.log_prob(z, h)

    def training_loss(
        self,
        x: Tensor,
        z: Tensor,
    ) -> Tensor:
        return -self.log_prob(x, z)

    def rsample(
        self,
        x: Tensor,
        n_samples: int,
    ) -> Tensor:
        h = self.encoder(x)
        return self.flow.rsample(h, n_samples)
