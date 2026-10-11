import pytest
import torch
from torch import nn
from torch.autograd.functional import jacobian
from zuko.transforms import MonotonicRQSTransform

from cf_npe.networks.posterior import RQSTransform
from cf_npe.networks.unified_nsf import UnifiedNSFPosterior, _layer_masks

DIM_X = 6
DIM_Z = 8
HEAD = 16
BATCH = 32


def _encoder(dim_in: int) -> nn.Module:
    return nn.Sequential(nn.Linear(dim_in, HEAD), nn.ReLU(), nn.Linear(HEAD, HEAD)).cuda()


def _flow(dim_mu: int, dim_z: int = DIM_Z) -> UnifiedNSFPosterior:
    return UnifiedNSFPosterior(
        encoder=_encoder(DIM_X + dim_mu),
        head_dim=HEAD,
        dim_x=DIM_X,
        dim_mu=dim_mu,
        dim_z=dim_z,
        hidden_dim=32,
        n_transforms=6,
        n_bins=8,
        tail_bound=7.0,
        n_layers=2,
        dropout=0.0,
    ).cuda()


@pytest.mark.parametrize("dim_mu", [0, DIM_Z])
def test_scores_samples_roundtrips(dim_mu):
    """forward/inverse invert, their log-dets cancel, and the inverse log-det matches autograd."""
    flow = _flow(dim_mu).double()
    packed = torch.randn(BATCH, DIM_X + dim_mu, device="cuda", dtype=torch.float64)
    z = torch.randn(BATCH, DIM_Z, device="cuda", dtype=torch.float64)

    h, mu = flow.encoder(packed), packed[:, DIM_X:]
    fwd, ld_f = flow._forward_flow(z, h, mu)
    rec, ld_i = flow._inverse_flow(fwd, h, mu)
    assert (rec - z).abs().max() < 1e-6
    assert (ld_f + ld_i).abs().max() < 1e-6

    h0, mu0 = h[:1], mu[:1]

    def inverse_map(z_single):
        z_base, _ = flow._inverse_flow(z_single.unsqueeze(0), h0, mu0)
        return z_base.squeeze(0)

    _, ld = flow._inverse_flow(z[:1], h0, mu0)
    jac = jacobian(inverse_map, z[0])
    expected = torch.linalg.slogdet(jac).logabsdet
    assert torch.allclose(ld.squeeze(0), expected, atol=1e-7), (
        f"inverse ladj={ld.item():.9f}, autograd slogdet(J)={expected.item():.9f}"
    )


def test_log_prob_integrates_to_one():
    """The density integrates to one on [-9,9]^2 for Standard NPE and CF-NPE."""
    dim_z = 2
    bound = 9.0
    n = 600
    for dim_mu in (0, dim_z):
        torch.manual_seed(0)
        flow = _flow(dim_mu, dim_z=dim_z).double().eval()
        packed = torch.randn(1, DIM_X + dim_mu, device="cuda", dtype=torch.float64)
        grid = torch.linspace(-bound, bound, n, device="cuda", dtype=torch.float64)
        dz = (grid[1] - grid[0]).item()
        gx, gy = torch.meshgrid(grid, grid, indexing="ij")
        z = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        lp = flow.log_prob(packed.expand(z.shape[0], -1), z)
        integral = lp.exp().sum() * dz * dz
        assert abs(integral.item() - 1.0) < 1e-3, f"dim_mu={dim_mu}: integral={integral.item():.6f}"


@pytest.mark.parametrize("dim_mu", [0, DIM_Z])
def test_inverse_flow_log_det_matches_autograd(dim_mu):
    """log_det from _inverse_flow equals autograd's slogdet(J) of the inverse map."""
    flow = _flow(dim_mu).double()
    packed = torch.randn(1, DIM_X + dim_mu, device="cuda", dtype=torch.float64)
    z = torch.randn(1, DIM_Z, device="cuda", dtype=torch.float64)
    h, mu = flow.encoder(packed), packed[:, DIM_X:]

    def inverse_map(z_single):
        z_base, _ = flow._inverse_flow(z_single.unsqueeze(0), h, mu)
        return z_base.squeeze(0)

    _, log_det = flow._inverse_flow(z, h, mu)
    jac = jacobian(inverse_map, z.squeeze(0))
    expected = torch.linalg.slogdet(jac).logabsdet
    assert torch.allclose(log_det.squeeze(0), expected, atol=1e-7), (
        f"log_det={log_det.item():.9f}, autograd slogdet(J)={expected.item():.9f}"
    )


def test_layer_masks_structure():
    """Each coordinate is active in n_transforms/2 masks; consecutive masks are complements."""
    dim_z, n_transforms = DIM_Z, 6
    masks = _layer_masks(dim_z, n_transforms)
    again = _layer_masks(dim_z, n_transforms)

    stacked = torch.stack(masks)
    assert (stacked.sum(0) == n_transforms // 2).all()
    for i in range(0, n_transforms, 2):
        assert (masks[i] == ~masks[i + 1]).all(), f"layers {i},{i + 1} are not complements"
    for mask in masks:
        assert mask.any() and not mask.all()
    for a, b in zip(masks, again, strict=True):
        assert (a == b).all(), "masks must be deterministic across calls"


def test_affine_mu_active_only_with_mu():
    """affine-mu_hat modulation is on for CF-NPE and absent for Standard NPE."""
    assert all(layer.affine_mu for layer in _flow(DIM_Z).couplings)
    assert all(not layer.affine_mu for layer in _flow(0).couplings)


def test_rqs_inverse_finite_at_extreme_knot_slopes():
    """Near a knot with extreme slopes, where zuko's inverse is NaN, ours is finite; elsewhere
    it matches zuko up to rounding."""
    gen = torch.Generator().manual_seed(0)
    n, n_bins = 20_000, 16
    widths = torch.randn(n, n_bins, generator=gen)
    heights = torch.randn(n, n_bins, generator=gen)
    derivs = 50.0 * torch.sign(torch.rand(n, n_bins - 1, generator=gen) - 0.5)
    zuko_rqs = MonotonicRQSTransform(widths, heights, derivs, bound=7.0)
    rqs = RQSTransform(widths, heights, derivs, bound=7.0)
    knot = torch.randint(1, n_bins, (n, 1), generator=gen)
    y_knot = torch.nextafter(zuko_rqs.vertical.gather(-1, knot).squeeze(-1), torch.tensor(-1e9))
    y = 14 * torch.rand(n, generator=gen) - 7

    assert torch.isnan(zuko_rqs.inv(y_knot)).any()
    assert torch.isfinite(rqs.inv(y_knot)).all()
    torch.testing.assert_close(rqs.inv(y), zuko_rqs.inv(y), rtol=0, atol=1e-3)
