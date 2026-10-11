"""Coupling neural spline flow shared by Standard NPE (``dim_mu == 0``) and CF-NPE Stage-2.

For CF-NPE (``dim_mu == dim_z``) the Stage-1 prediction mu_hat enters through the context
encoder's input and a per-layer affine modulation of the spline parameters.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from cf_npe.networks.models import unpack_xmu
from cf_npe.networks.posterior import (
    RQSTransform,
    SamplingMomentMixin,
    _spline_params_per_dim,
    mlp_layers,
)

_EMBED_DIM = 16


def _apply_rqs(
    z: Tensor,
    params: Tensor,
    n_bins: int,
    tail_bound: float,
    inverse: bool,
) -> tuple[Tensor, Tensor]:
    widths, heights, derivs = params.split([n_bins, n_bins, n_bins - 1], dim=-1)
    transform = RQSTransform(widths, heights, derivs, bound=tail_bound)
    if inverse:
        z_out = transform.inv(z)
        log_det = -transform.log_abs_det_jacobian(z_out, z)
    else:
        z_out = transform(z)
        log_det = transform.log_abs_det_jacobian(z, z_out)
    return z_out, log_det.sum(dim=-1)


def _layer_masks(dim_z: int, n_transforms: int) -> list[Tensor]:
    """Per-layer boolean active masks over the dim_z coordinates.

    A fixed-seed random half-partition per layer pair, alternated with its complement. At
    dim_z == 1 the single coordinate is active in every layer; its spline parameters then come
    from the conditioning context alone.
    """
    masks: list[Tensor] = []
    for layer in range(n_transforms):
        g = torch.Generator().manual_seed(20260519 + layer // 2)
        half = torch.randperm(dim_z, generator=g)[: dim_z // 2]
        mask = torch.zeros(dim_z, dtype=torch.bool)
        mask[half] = True
        if layer % 2 == 1:
            mask = ~mask
        if not mask.any():
            mask[0] = True
        masks.append(mask)
    return masks


class _PerDimConditioner(nn.Module):
    """Weight-tied per-coordinate MLP: ``(z_passive, h, embed_i) -> params_i``.

    The learned embedding ``embed_i`` identifies the coordinate, so the shared weights can
    produce different splines per coordinate with or without mu_hat.
    """

    def __init__(
        self,
        n_passive: int,
        head_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_bins: int,
        dropout: float,
    ):
        super().__init__()
        in_dim = n_passive + head_dim + _EMBED_DIM
        self.head = nn.Sequential(
            *mlp_layers(in_dim, hidden_dim, n_layers, nn.ReLU, dropout),
            nn.Linear(hidden_dim, _spline_params_per_dim(n_bins)),
        )

    def forward(self, z_passive: Tensor, h: Tensor, embed_active: Tensor) -> Tensor:
        batch = z_passive.shape[:-1]
        n_active = embed_active.shape[0]
        z_b = z_passive.unsqueeze(-2).expand(*batch, n_active, z_passive.shape[-1])
        h_b = h.unsqueeze(-2).expand(*batch, n_active, h.shape[-1])
        e_b = embed_active.reshape((1,) * len(batch) + embed_active.shape).expand(
            *batch, n_active, embed_active.shape[-1]
        )
        return self.head(torch.cat([z_b, h_b, e_b], dim=-1))


class _CouplingLayer(nn.Module):
    """Spline-warp the active coords and copy the rest.

    ``affine_mu`` scales the spline params by ``(1 + Linear(mu_hat))`` (CF-NPE only).
    """

    active_idx: Tensor
    passive_idx: Tensor

    def __init__(
        self,
        dim_z: int,
        head_dim: int,
        mask: Tensor,
        affine_mu: bool,
        hidden_dim: int,
        n_layers: int,
        n_bins: int,
        tail_bound: float,
        dropout: float,
    ):
        super().__init__()
        self.register_buffer("active_idx", torch.where(mask)[0])
        self.register_buffer("passive_idx", torch.where(~mask)[0])
        self.n_bins = n_bins
        self.tail_bound = tail_bound
        self.affine_mu = affine_mu
        n_active = int(mask.sum().item())
        n_passive = dim_z - n_active
        self.conditioner = _PerDimConditioner(
            n_passive, head_dim, hidden_dim, n_layers, n_bins, dropout
        )
        if affine_mu:
            self.affine_proj = nn.Linear(1, _spline_params_per_dim(n_bins))
            nn.init.zeros_(self.affine_proj.weight)
            nn.init.zeros_(self.affine_proj.bias)

    def forward(
        self,
        z: Tensor,
        h: Tensor,
        mu: Tensor,
        embed: Tensor,
        inverse: bool,
    ) -> tuple[Tensor, Tensor]:
        z_passive = z.index_select(-1, self.passive_idx)
        z_active = z.index_select(-1, self.active_idx)
        embed_active = embed.index_select(0, self.active_idx)
        params = self.conditioner(z_passive, h, embed_active)
        if self.affine_mu:
            mu_active = mu.index_select(-1, self.active_idx)
            params = params * (1.0 + self.affine_proj(mu_active.unsqueeze(-1)))
        z_active_new, log_det = _apply_rqs(z_active, params, self.n_bins, self.tail_bound, inverse)
        z_out = torch.empty_like(z)
        z_out = z_out.index_copy(-1, self.passive_idx, z_passive)
        z_out = z_out.index_copy(-1, self.active_idx, z_active_new)
        return z_out, log_det


class UnifiedNSFPosterior(SamplingMomentMixin, nn.Module):
    """Conditional coupling NSF over input packed as ``cat[x, mu_hat]``, shape (B, dim_x + dim_mu).

    ``dim_mu`` is 0 for Standard NPE and ``dim_z`` for CF-NPE Stage-2.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head_dim: int,
        dim_x: int,
        dim_mu: int,
        dim_z: int,
        hidden_dim: int,
        n_transforms: int,
        n_bins: int,
        tail_bound: float,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        if dim_mu not in (0, dim_z):
            raise ValueError(f"dim_mu must be 0 or dim_z; got {dim_mu} vs dim_z={dim_z}")
        affine_mu = dim_mu > 0
        self.dim_x = dim_x
        self.dim_mu = dim_mu
        self.dim_z = dim_z
        self.encoder = encoder
        self.dim_embed = nn.Parameter(torch.randn(dim_z, _EMBED_DIM) * 0.1)
        self.couplings = nn.ModuleList(
            [
                _CouplingLayer(
                    dim_z=dim_z,
                    head_dim=head_dim,
                    mask=mask,
                    affine_mu=affine_mu,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    n_bins=n_bins,
                    tail_bound=tail_bound,
                    dropout=dropout,
                )
                for mask in _layer_masks(dim_z, n_transforms)
            ]
        )

    def _forward_flow(self, z: Tensor, h: Tensor, mu: Tensor) -> tuple[Tensor, Tensor]:
        log_det = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        for layer in self.couplings:
            z, ld = layer(z, h, mu, self.dim_embed, inverse=False)
            log_det = log_det + ld
        return z, log_det

    def _inverse_flow(self, z: Tensor, h: Tensor, mu: Tensor) -> tuple[Tensor, Tensor]:
        log_det = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        for layer in reversed(self.couplings):
            z, ld = layer(z, h, mu, self.dim_embed, inverse=True)
            log_det = log_det + ld
        return z, log_det

    def log_prob(self, packed: Tensor, z: Tensor) -> Tensor:
        _, mu = unpack_xmu(packed, self.dim_x)
        z_base, log_det = self._inverse_flow(z, self.encoder(packed), mu)
        base_logp = -0.5 * (z_base.pow(2).sum(dim=-1) + self.dim_z * math.log(2 * math.pi))
        return base_logp + log_det

    def training_loss(self, packed: Tensor, z: Tensor) -> Tensor:
        return -self.log_prob(packed, z)

    def rsample(self, x: Tensor, n_samples: int) -> Tensor:
        _, mu = unpack_xmu(x, self.dim_x)
        h = self.encoder(x)
        batch = x.shape[0]
        z0 = torch.randn(batch, n_samples, self.dim_z, device=x.device)
        h_b = h.unsqueeze(1).expand(-1, n_samples, -1)
        mu_b = mu.unsqueeze(1).expand(-1, n_samples, -1)
        z_data, _ = self._forward_flow(
            z0.reshape(batch * n_samples, self.dim_z),
            h_b.reshape(batch * n_samples, h.shape[-1]),
            mu_b.reshape(batch * n_samples, self.dim_mu),
        )
        return z_data.view(batch, n_samples, self.dim_z)
