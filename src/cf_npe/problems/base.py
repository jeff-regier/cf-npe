"""Base class for simulation problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from cf_npe.networks.posterior import GaussianPosterior, PosteriorModel, mlp_encoder
from cf_npe.networks.unified_nsf import UnifiedNSFPosterior

if TYPE_CHECKING:
    from cf_npe.stages.stage2 import Stage2Config


@dataclass
class EncoderConfig:
    hidden_dim: int
    n_layers: int
    dropout: float


@dataclass
class NSFConfig:
    """Spline-flow hyperparameters for the ``nsf`` posterior."""

    n_transforms: int
    n_bins: int
    tail_bound: float


@dataclass
class Problem(ABC):
    dim: int

    @property
    def dim_z(self) -> int:
        return self.dim

    @property
    def dim_x(self) -> int:
        return self.dim

    @abstractmethod
    def sample_prior(self, n: int) -> Tensor:
        pass

    @abstractmethod
    def sample_likelihood(self, z: Tensor) -> Tensor:
        pass

    def sample_joint(self, n: int) -> tuple[Tensor, Tensor]:
        z = self.sample_prior(n)
        x = self.sample_likelihood(z)
        return x, z

    def create_encoder(self, encoder_cfg: EncoderConfig) -> nn.Module:
        return mlp_encoder(
            self.dim_x,
            encoder_cfg.hidden_dim,
            encoder_cfg.n_layers,
            encoder_cfg.dropout,
        )

    def encoder_head_dim(self, encoder_cfg: EncoderConfig) -> int:
        return encoder_cfg.hidden_dim

    def create_mean_head(self, head_dim: int, dim_z: int) -> nn.Module:
        """Stage-1 MeanRegressor readout head on encoder features."""
        return nn.Linear(head_dim, dim_z)

    def arch_fingerprint(self) -> dict | None:
        """Stage-1 architecture identity checked on Stage-1 cache load.

        None when the cache's own ``encoder_cfg`` rebuilds the Stage-1 networks, so there is
        nothing to match.
        """
        return None

    def create_offset_posterior(self, cfg: Stage2Config) -> nn.Module:
        """Untrained CF-NPE Stage-2 flow on packed ``(x, mu)``: a UnifiedNSFPosterior on an
        MLP encoder."""
        return UnifiedNSFPosterior(
            encoder=mlp_encoder(self.dim_x + self.dim_z, cfg.hidden_dim, cfg.n_layers, cfg.dropout),
            head_dim=cfg.hidden_dim,
            dim_x=self.dim_x,
            dim_mu=self.dim_z,
            dim_z=self.dim_z,
            hidden_dim=cfg.hidden_dim,
            n_transforms=cfg.n_transforms,
            n_bins=cfg.n_bins,
            tail_bound=cfg.tail_bound,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        )

    def create_posterior(
        self,
        encoder_cfg: EncoderConfig,
        posterior_type: str,
        flow_cfg: NSFConfig | None,
    ) -> PosteriorModel:
        """Posterior on ``create_encoder``'s encoder; ``flow_cfg`` is required for ``nsf`` only."""
        encoder = self.create_encoder(encoder_cfg)
        head_dim = self.encoder_head_dim(encoder_cfg)
        if posterior_type == "gaussian":
            model = GaussianPosterior(
                encoder=encoder,
                head_dim=head_dim,
                dim_z=self.dim_z,
            )
        elif posterior_type == "nsf":
            nsf = cast(NSFConfig, flow_cfg)
            model = UnifiedNSFPosterior(
                encoder=encoder,
                head_dim=head_dim,
                dim_x=self.dim_x,
                dim_mu=0,
                dim_z=self.dim_z,
                hidden_dim=encoder_cfg.hidden_dim,
                n_transforms=nsf.n_transforms,
                n_bins=nsf.n_bins,
                tail_bound=nsf.tail_bound,
                n_layers=encoder_cfg.n_layers,
                dropout=encoder_cfg.dropout,
            )
        else:
            raise ValueError(f"Unknown posterior_type: {posterior_type}")
        return cast(PosteriorModel, model.to("cuda"))

    def posterior_mean(self, x: Tensor) -> Tensor:
        """Analytic posterior mean, for problems that have one."""
        raise NotImplementedError

    def posterior_std(self) -> float:
        """Analytic posterior std, for problems that have one."""
        raise NotImplementedError
