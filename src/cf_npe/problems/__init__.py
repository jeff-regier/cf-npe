"""Problems, exported for Hydra's ``cf_npe.problems.<Class>`` targets."""

from cf_npe.problems.simple import Funnel, Gaussian, Mixture
from cf_npe.problems.strong_lensing import StrongLensing

__all__ = ["Funnel", "Gaussian", "Mixture", "StrongLensing"]
