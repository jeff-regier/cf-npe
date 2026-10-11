"""Strong-lensing pixelated source reconstruction from the lensed image alone.

z is the flattened source-plane pixel grid; physical surface brightness is ``z * source_amp``.
The SIE + external-shear lens parameters are drawn per sample and not observed.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from cf_npe.networks.local_nsf import LocalNSFPosterior, ResNet2dConditionerConfig
from cf_npe.networks.models import unpack_xmu
from cf_npe.networks.posterior import PosteriorModel
from cf_npe.networks.resnet2d import PaddingMode, ResNet2DDown, ResNet2DSame
from cf_npe.problems.base import EncoderConfig, NSFConfig, Problem

if TYPE_CHECKING:
    from cf_npe.stages.stage2 import Stage2Config


@functools.cache
def _load_simulation(data_path: str) -> dict:
    """The simulation file at ``data_path``, read once per process; the training and test
    problems slice their pools from it."""
    return torch.load(Path(data_path), weights_only=False)


class _ImageEncoder(nn.Module):
    """Image encoder that exposes its output channel count."""

    out_channels: int


class _ResNetImageEncoder(_ImageEncoder):
    """Downsampling ResNet on the flattened image; ``asinh_softening`` s > 0 applies asinh(x/s)."""

    def __init__(
        self,
        image_resolution: int,
        n_downsamples: int,
        dropout: float,
        asinh_softening: float,
        channel_base: int,
        blocks_per_level: int,
        activation: str,
        norm_groups: int,
        padding_mode: PaddingMode,
        use_se: bool,
        drop_path_max: float,
    ):
        super().__init__()
        self.image_resolution = image_resolution
        self.asinh_softening = asinh_softening
        self.resnet = ResNet2DDown(
            in_channels=1,
            base_channels=channel_base,
            n_downsamples=n_downsamples,
            blocks_per_level=blocks_per_level,
            dropout=dropout,
            activation=activation,
            norm_groups=norm_groups,
            padding_mode=padding_mode,
            use_se=use_se,
            drop_path_max=drop_path_max,
        )
        self.out_channels = self.resnet.out_channels

    def forward(self, x_flat: Tensor) -> Tensor:
        batch = x_flat.shape[0]
        h = self.image_resolution
        x = x_flat.view(batch, 1, h, h)
        if self.asinh_softening > 0.0:
            x = torch.asinh(x / self.asinh_softening)
        return self.resnet(x)


class _ResNetMeanHead(nn.Module):
    """Per-pixel mean from (B, in_channels, n, n) encoder features, flattened to (B, n*n)."""

    def __init__(
        self,
        in_channels: int,
        n_side: int,
        width: int,
        n_blocks: int,
        n_attention: int,
        attention_heads: int,
        dropout: float,
        activation: str,
        norm_groups: int,
        padding_mode: PaddingMode,
        use_se: bool,
        drop_path_max: float,
        use_pos_encoding: bool,
        use_global_pool: bool,
    ):
        super().__init__()
        self.net = ResNet2DSame(
            in_channels=in_channels,
            out_channels=1,
            width=width,
            n_blocks=n_blocks,
            dropout=dropout,
            n_attention=n_attention,
            attention_heads=attention_heads,
            n_side=n_side,
            activation=activation,
            norm_groups=norm_groups,
            padding_mode=padding_mode,
            use_se=use_se,
            drop_path_max=drop_path_max,
            use_pos_encoding=use_pos_encoding,
            use_global_pool=use_global_pool,
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h).flatten(start_dim=1)


class _MuConcatOffsetEncoder(_ImageEncoder):
    """Concatenate ``inner(x)`` with a conv lift of mu on the source grid: (B, C_x + C_mu, n, n)."""

    def __init__(
        self,
        inner: _ImageEncoder,
        dim_x_raw: int,
        n_side: int,
        mu_proj_channels: int,
    ):
        super().__init__()
        self.inner = inner
        self.dim_x_raw = dim_x_raw
        self.n_side = n_side
        self.mu_lift = nn.Sequential(
            nn.Conv2d(1, mu_proj_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mu_proj_channels, mu_proj_channels, 3, padding=1),
        )
        self.out_channels = inner.out_channels + mu_proj_channels

    def forward(self, cond: Tensor) -> Tensor:
        x_raw, mu_flat = unpack_xmu(cond, self.dim_x_raw)
        feat_x = self.inner(x_raw)
        feat_mu = self.mu_lift(mu_flat.view(-1, 1, self.n_side, self.n_side))
        return torch.cat([feat_x, feat_mu], dim=1)


@dataclass
class SLEncoderArch:
    """Stage-1 image encoder (ResNet2DDown) architecture."""

    base_channels: int = 32
    blocks_per_level: int = 1
    dropout: float = 0.1
    asinh_softening: float = 1.0
    activation: str = "gelu"
    norm_groups: int = 32
    padding_mode: PaddingMode = "zeros"
    use_se: bool = False
    drop_path_max: float = 0.0


@dataclass
class SLMeanHeadArch:
    """Stage-1 mean head (ResNet2DSame) architecture."""

    width: int = 128
    n_blocks: int = 6
    n_attention: int = 1
    attention_heads: int = 4
    dropout: float = 0.1
    use_pos_encoding: bool = False
    use_global_pool: bool = False
    activation: str = "gelu"
    norm_groups: int = 32
    padding_mode: PaddingMode = "zeros"
    use_se: bool = False
    drop_path_max: float = 0.0


@dataclass
class SLVelocityHeadArch:
    """NSF posterior and Stage-2 offset-encoder architecture.

    ``data_encoding_dim`` / ``mu_proj_channels`` set the offset encoder; the rest set the flow.
    """

    data_encoding_dim: int = 256
    mu_proj_channels: int = 8
    n_transforms: int = 6
    n_bins: int = 8
    tail_bound: float = 10.0
    mask_kind: str = "stripe_row"
    nsf_conditioner_width: int = 128
    nsf_n_blocks: int = 6
    nsf_n_attention: int = 1
    nsf_attention_heads: int = 4
    nsf_dropout: float = 0.0
    nsf_use_se: bool = False
    nsf_drop_path_max: float = 0.0
    nsf_use_pos_encoding: bool = True
    nsf_use_global_pool: bool = True


@dataclass
class StrongLensing(Problem):
    """Pixelated-source strong lensing with a lens drawn per sample.

    ``dim`` is unused: dim_z = source_resolution**2 and dim_x = image_resolution**2.
    """

    dim: int = 256
    source_resolution: int = 16
    image_resolution: int = 64
    noise_sigma: float = 0.05
    data_path: str = "data/strong_lensing/source16_varlens.pt"
    split_seed: int = 0
    is_test: bool = False
    test_fraction: float = 0.1
    # Used instead of the shared EncoderConfig. The Stage-1 cache load checks that encoder and
    # mean_head match the cached architecture (arch_fingerprint).
    encoder: SLEncoderArch = field(default_factory=SLEncoderArch)
    mean_head: SLMeanHeadArch = field(default_factory=SLMeanHeadArch)
    velocity_head: SLVelocityHeadArch = field(default_factory=SLVelocityHeadArch)

    _z_pool: Tensor = field(init=False, repr=False, compare=False)
    _y_pool: Tensor = field(init=False, repr=False, compare=False)
    _gp_chol: Tensor = field(init=False, repr=False, compare=False)
    # Stride-2 stages taking the image down to the source grid.
    _n_downsamples: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        ratio = self.image_resolution // self.source_resolution
        self._n_downsamples = int(math.log2(ratio))
        if 2**self._n_downsamples != ratio:
            raise ValueError(f"image/source resolution ratio must be a power of 2; got {ratio}")
        payload = _load_simulation(self.data_path)
        cached_src_res = int(payload["source_resolution"])
        cached_img_res = int(payload["image_resolution"])
        if cached_src_res != self.source_resolution:
            raise ValueError(
                f"data_path source_resolution={cached_src_res} but "
                f"problem source_resolution={self.source_resolution}"
            )
        if cached_img_res != self.image_resolution:
            raise ValueError(
                f"data_path image_resolution={cached_img_res} but "
                f"problem image_resolution={self.image_resolution}"
            )
        self._gp_chol = payload["gp_chol"].float().to("cuda")
        self._setup_pools(payload["z"], payload["y_clean"])

    def _pool_indices(self, n_total: int) -> Tensor:
        gen = torch.Generator().manual_seed(self.split_seed)
        perm = torch.randperm(n_total, generator=gen)
        n_test = int(self.test_fraction * n_total)
        return perm[:n_test] if self.is_test else perm[n_test:]

    def _setup_pools(self, z_full: Tensor, y_full: Tensor) -> None:
        idx = self._pool_indices(z_full.shape[0])
        self._z_pool = z_full[idx].float().to("cuda")
        self._y_pool = y_full[idx].float().to("cuda")

    @property
    def dim_z(self) -> int:
        return self.source_resolution * self.source_resolution

    @property
    def dim_x(self) -> int:
        return self.image_resolution * self.image_resolution

    def sample_prior(self, n: int) -> Tensor:
        eta = torch.randn(n, self.dim_z, device="cuda")
        return eta @ self._gp_chol.T

    def sample_joint(self, n: int) -> tuple[Tensor, Tensor]:
        pool_size = self._z_pool.shape[0]
        if n > pool_size:
            raise ValueError(f"requested n={n} > pool_size={pool_size}")
        idx = torch.randperm(pool_size, device="cuda")[:n]
        z = self._z_pool[idx]
        y_clean = self._y_pool[idx]
        y_noisy = y_clean + self.noise_sigma * torch.randn_like(y_clean)
        return y_noisy.reshape(n, -1), z

    def sample_likelihood(self, z: Tensor) -> Tensor:
        raise NotImplementedError(
            "the lens is drawn per sample and not exposed; use sample_joint, which draws "
            "(x, z) pairs from the pre-rendered pool"
        )

    def _image_encoder(self, channel_base: int) -> _ResNetImageEncoder:
        e = self.encoder
        return _ResNetImageEncoder(
            image_resolution=self.image_resolution,
            n_downsamples=self._n_downsamples,
            asinh_softening=e.asinh_softening,
            dropout=e.dropout,
            channel_base=channel_base,
            blocks_per_level=e.blocks_per_level,
            activation=e.activation,
            norm_groups=e.norm_groups,
            padding_mode=e.padding_mode,
            use_se=e.use_se,
            drop_path_max=e.drop_path_max,
        )

    def create_encoder(self, encoder_cfg: EncoderConfig) -> _ImageEncoder:
        del encoder_cfg  # architecture comes from self.encoder
        return self._image_encoder(self.encoder.base_channels)

    def encoder_head_dim(self, encoder_cfg: EncoderConfig) -> int:
        del encoder_cfg
        return self.encoder.base_channels * 2**self._n_downsamples

    def arch_fingerprint(self) -> dict:
        def _no_dropout(cfg: SLEncoderArch | SLMeanHeadArch) -> dict:
            d = asdict(cfg)
            del d["dropout"]
            return d

        return {"encoder": _no_dropout(self.encoder), "mean_head": _no_dropout(self.mean_head)}

    def create_mean_head(self, head_dim: int, dim_z: int) -> nn.Module:
        del dim_z
        m = self.mean_head
        return _ResNetMeanHead(
            in_channels=head_dim,
            n_side=self.source_resolution,
            width=m.width,
            n_blocks=m.n_blocks,
            n_attention=m.n_attention,
            attention_heads=m.attention_heads,
            dropout=m.dropout,
            activation=m.activation,
            norm_groups=m.norm_groups,
            padding_mode=m.padding_mode,
            use_se=m.use_se,
            drop_path_max=m.drop_path_max,
            use_pos_encoding=m.use_pos_encoding,
            use_global_pool=m.use_global_pool,
        )

    def _local_nsf(self, encoder: _ImageEncoder) -> LocalNSFPosterior:
        """``LocalNSFPosterior`` on the source grid with ``self.velocity_head``'s flow."""
        v = self.velocity_head
        return LocalNSFPosterior(
            encoder=encoder,
            head_dim=encoder.out_channels,
            hidden_dim=v.nsf_conditioner_width,
            n_transforms=v.n_transforms,
            n_bins=v.n_bins,
            n_side=self.source_resolution,
            tail_bound=v.tail_bound,
            mask_kind=v.mask_kind,
            resnet_cfg=ResNet2dConditionerConfig(
                n_blocks=v.nsf_n_blocks,
                n_attention=v.nsf_n_attention,
                attention_heads=v.nsf_attention_heads,
                dropout=v.nsf_dropout,
                use_se=v.nsf_use_se,
                drop_path_max=v.nsf_drop_path_max,
                use_pos_encoding=v.nsf_use_pos_encoding,
                use_global_pool=v.nsf_use_global_pool,
            ),
        )

    def create_offset_posterior(self, cfg: Stage2Config) -> nn.Module:
        """The flow on a fresh trainable image encoder combined with mu_hat
        (``_MuConcatOffsetEncoder``); ``cfg``'s architecture fields are unused."""
        del cfg
        v = self.velocity_head
        denom = 2**self._n_downsamples
        if v.data_encoding_dim % denom != 0:
            raise ValueError(
                f"velocity_head.data_encoding_dim ({v.data_encoding_dim}) "
                f"must be divisible by 2^n_downsamples ({denom})"
            )
        encoder = _MuConcatOffsetEncoder(
            inner=self._image_encoder(v.data_encoding_dim // denom),
            dim_x_raw=self.dim_x,
            n_side=self.source_resolution,
            mu_proj_channels=v.mu_proj_channels,
        )
        return self._local_nsf(encoder)

    def create_posterior(
        self,
        encoder_cfg: EncoderConfig,
        posterior_type: str,
        flow_cfg: NSFConfig | None,
    ) -> PosteriorModel:
        """The flow on the Stage-1 image encoder; ``posterior_type`` must be ``'nsf'``.

        Flow settings come from ``self.velocity_head``; ``flow_cfg`` is unused.
        """
        if posterior_type != "nsf":
            raise ValueError(
                f"strong lensing supports posterior_type='nsf' only; got {posterior_type!r}"
            )
        del flow_cfg
        model = self._local_nsf(self.create_encoder(encoder_cfg))
        return cast(PosteriorModel, model.to("cuda"))
