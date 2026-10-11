"""Mask alternation, log-det Jacobian, and invertibility of the strong-lensing coupling flow."""

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor
from torch.autograd.functional import jacobian

from cf_npe.networks.local_nsf import (
    CouplingLayer,
    JointNSF,
    ResNet2dConditioner,
    ResNet2dConditionerConfig,
    _build_level_indices,
)

DEVICE = torch.device("cuda")
N_SIDE = 3
DIM_Z = N_SIDE * N_SIDE
CTX = 4
BATCH = 4
RESNET_CFG = ResNet2dConditionerConfig(
    n_blocks=2,
    n_attention=1,
    attention_heads=4,
    dropout=0.0,
    use_se=False,
    drop_path_max=0.0,
    use_pos_encoding=False,
    use_global_pool=False,
)


def _flow(n_transforms: int = 3) -> JointNSF:
    return (
        JointNSF(
            context_dim=CTX,
            n_side=N_SIDE,
            resnet_cfg=RESNET_CFG,
            n_transforms=n_transforms,
            n_bins=4,
            hidden_dim=8,
            tail_bound=5.0,
            mask_kind="stripe_row",
        )
        .to(DEVICE)
        .eval()
    )


def _randomize_readouts(flow: JointNSF, std: float) -> None:
    """Randomize the (else zero-init) readouts so the couplings are non-trivial."""
    for layer in cast(list[CouplingLayer], flow.layers):
        readout = cast(torch.nn.Conv2d, cast(ResNet2dConditioner, layer.conditioner).net.readout)
        torch.nn.init.normal_(readout.weight, std=std)
        torch.nn.init.normal_(cast(Tensor, readout.bias), std=std)


def test_index_coverage():
    for even_fixed in (True, False):
        fixed, transform = _build_level_indices(N_SIDE, even_fixed, "stripe_row")
        all_idx = torch.cat([fixed, transform]).sort().values
        assert torch.equal(all_idx, torch.arange(DIM_Z))


def test_mask_alternates_across_layers():
    flow = _flow(n_transforms=4)
    fixed_sets = [set(cast(CouplingLayer, layer).fixed_idx.tolist()) for layer in flow.layers]
    assert fixed_sets[0] == fixed_sets[2]
    assert fixed_sets[1] == fixed_sets[3]
    assert fixed_sets[0].isdisjoint(fixed_sets[1])
    assert fixed_sets[0] | fixed_sets[1] == set(range(DIM_Z))


def _forward_z(layer: CouplingLayer, ctx: Tensor) -> Callable[[Tensor], Tensor]:
    def f(z: Tensor) -> Tensor:
        z_out, _ = layer.forward_and_ladj(z.unsqueeze(0), ctx.unsqueeze(0))
        return z_out.squeeze(0)

    return f


def test_coupling_ladj_matches_autograd_jacobian():
    torch.manual_seed(0)
    flow = _flow().double()
    _randomize_readouts(flow, std=0.5)
    layer = cast(CouplingLayer, flow.layers[0])
    z = torch.randn(DIM_Z, device=DEVICE, dtype=torch.float64)
    ctx = torch.randn(CTX, N_SIDE, N_SIDE, device=DEVICE, dtype=torch.float64)
    _, ladj = layer.forward_and_ladj(z.unsqueeze(0), ctx.unsqueeze(0))
    expected = torch.linalg.slogdet(jacobian(_forward_z(layer, ctx), z)).logabsdet
    assert not torch.allclose(expected, torch.zeros_like(expected), atol=1e-3)
    assert torch.allclose(ladj.squeeze(0), expected, atol=1e-5)


def test_joint_ladj_matches_autograd_jacobian():
    torch.manual_seed(1)
    flow = _flow().double()
    _randomize_readouts(flow, std=0.5)
    layers = cast(list[CouplingLayer], flow.layers)
    z = torch.randn(DIM_Z, device=DEVICE, dtype=torch.float64)
    ctx = torch.randn(1, CTX, N_SIDE, N_SIDE, device=DEVICE, dtype=torch.float64)

    def forward_all(z_in: Tensor) -> Tensor:
        out = z_in.unsqueeze(0)
        for layer in layers:
            out, _ = layer.forward_and_ladj(out, ctx)
        return out.squeeze(0)

    out = z.unsqueeze(0)
    total = torch.zeros((), device=DEVICE, dtype=torch.float64)
    for layer in layers:
        out, ladj = layer.forward_and_ladj(out, ctx)
        total = total + ladj.squeeze(0)
    expected = torch.linalg.slogdet(jacobian(forward_all, z)).logabsdet
    assert torch.allclose(total, expected, atol=1e-5)


def test_joint_nsf_invertibility():
    torch.manual_seed(2)
    flow = _flow().double()
    _randomize_readouts(flow, std=0.5)
    ctx = torch.randn(BATCH, CTX, N_SIDE, N_SIDE, device=DEVICE, dtype=torch.float64)
    z_orig = torch.randn(BATCH, DIM_Z, device=DEVICE, dtype=torch.float64)
    z = z_orig
    for layer in flow.layers:
        z, _ = cast(CouplingLayer, layer).forward_and_ladj(z, ctx)
    for layer in reversed(flow.layers):
        z = cast(CouplingLayer, layer).inverse(z, ctx)
    assert torch.allclose(z_orig, z, atol=1e-8)
