"""Generate (z, clean image, lens params) triples for the strong-lensing problem.

z is a unit-variance RBF Gaussian process on the R x R source grid and is used directly as
source surface brightness (no softplus), so the forward map is linear in z. SIE + external-shear
lens parameters are drawn uniformly per sample. Observation noise is added later, in
``StrongLensing.sample_joint``.

Usage (one file per seed; see the Makefile), with ``sl_n`` samples from conf/scale/:
    python -m cf_npe.problems.prepare_strong_lensing \
        --source-res 16 --gp-lengthscale 1.0 --seed 0 \
        --out out/data/strong_lensing/source16_varlens_seed0.pt
"""

import argparse
import time
from functools import cache, partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

from cf_npe import load_scale

LENS_MODEL_LIST: list[str] = ["SIE", "SHEAR"]

# Uniform ranges: Einstein radius theta_E in arcsec, SIE ellipticity (e1, e2), external shear
# (gamma1, gamma2).
LENS_PARAM_NAMES: list[str] = ["theta_E", "e1", "e2", "gamma1", "gamma2"]
LENS_PARAM_LO: np.ndarray = np.array([0.8, -0.25, -0.25, -0.1, -0.1], dtype=np.float32)
LENS_PARAM_HI: np.ndarray = np.array([1.2, 0.25, 0.25, 0.1, 0.1], dtype=np.float32)


def _build_gp_chol(source_res: int, lengthscale_pixels: float) -> np.ndarray:
    """L of shape (R*R, R*R) with L L^T = K, K_ij = exp(-||p_i - p_j||^2 / (2 ℓ^2)) in pixels."""
    coords = np.arange(source_res, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    pos = np.stack([yy.ravel(), xx.ravel()], axis=-1)  # (R*R, 2)
    sq = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=-1)
    K = np.exp(-sq / (2.0 * lengthscale_pixels**2))
    K += 1e-4 * np.eye(K.shape[0])
    return np.linalg.cholesky(K)


def _kwargs_lens_from_params(params: np.ndarray) -> list[dict]:
    theta_E, e1, e2, gamma1, gamma2 = map(float, params)
    return [
        {"theta_E": theta_E, "e1": e1, "e2": e2, "center_x": 0.0, "center_y": 0.0},
        {"gamma1": gamma1, "gamma2": gamma2, "ra_0": 0.0, "dec_0": 0.0},
    ]


PSF_FWHM_ARCSEC: float = 0.1
IMAGE_PIXEL_SCALE_ARCSEC: float = 0.05


@cache
def _image_model(image_res: int) -> ImageModel:
    """Lenstronomy image model, built once per worker process."""
    psf_class = PSF(
        psf_type="GAUSSIAN",
        fwhm=PSF_FWHM_ARCSEC,
        pixel_size=IMAGE_PIXEL_SCALE_ARCSEC,
    )
    half_extent = (image_res - 1) / 2.0 * IMAGE_PIXEL_SCALE_ARCSEC
    transform = np.array([[IMAGE_PIXEL_SCALE_ARCSEC, 0.0], [0.0, IMAGE_PIXEL_SCALE_ARCSEC]])
    data_class = ImageData(
        image_data=np.zeros((image_res, image_res)),
        background_rms=0.0,
        exposure_time=1.0,
        ra_at_xy_0=-half_extent,
        dec_at_xy_0=-half_extent,
        transform_pix2angle=transform,
    )
    lens_model_class = LensModel(lens_model_list=LENS_MODEL_LIST)
    source_model_class = LightModel(light_model_list=["INTERPOL"])
    return ImageModel(
        data_class=data_class,
        psf_class=psf_class,
        lens_model_class=lens_model_class,
        source_model_class=source_model_class,
        kwargs_numerics={"supersampling_factor": 1},
    )


def _simulate_chunk(
    chunk: tuple[np.ndarray, np.ndarray],
    image_res: int,
    source_res: int,
    source_amp: float,
    source_fov_arcsec: float,
) -> np.ndarray:
    """(n_chunk, image_dim) lensed images from z (n_chunk, dim_z) and lens params."""
    z_chunk, lens_chunk = chunk
    image_model = _image_model(image_res)
    source_scale = source_fov_arcsec / source_res
    n_chunk = z_chunk.shape[0]
    out = np.empty((n_chunk, image_res * image_res), dtype=np.float32)
    for i in range(n_chunk):
        # Negative surface brightness is unphysical but keeps the forward map linear in z.
        src_intensity = z_chunk[i].reshape(source_res, source_res).astype(np.float64)
        kwargs_source = [
            {
                "image": src_intensity,
                "amp": source_amp,
                "center_x": 0.0,
                "center_y": 0.0,
                "phi_G": 0.0,
                "scale": source_scale,
            }
        ]
        # lenstronomy's INTERPOL caches its spline on the first call and then ignores the
        # ``image`` arg; without this every sample in a worker gets the first source.
        image_model.SourceModel.delete_interpol_caches()
        image = image_model.image(_kwargs_lens_from_params(lens_chunk[i]), kwargs_source)
        out[i] = image.astype(np.float32).ravel()
    return out


def _split_into_chunks(
    z: np.ndarray, lens: np.ndarray, chunk: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    n = z.shape[0]
    return [(z[start : start + chunk], lens[start : start + chunk]) for start in range(0, n, chunk)]


def _generate_clean_images(
    z: np.ndarray,
    lens_params: np.ndarray,
    image_res: int,
    source_res: int,
    source_amp: float,
    source_fov_arcsec: float,
    chunk: int,
) -> np.ndarray:
    """(n, image_dim) lensed images."""
    simulate = partial(
        _simulate_chunk,
        image_res=image_res,
        source_res=source_res,
        source_amp=source_amp,
        source_fov_arcsec=source_fov_arcsec,
    )
    with Pool() as pool:
        chunks = _split_into_chunks(z, lens_params, chunk)
        return np.concatenate(list(pool.imap(simulate, chunks)), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--source-res", type=int, required=True)
    parser.add_argument("--image-res", type=int, default=64)
    parser.add_argument(
        "--source-fov",
        type=float,
        default=1.0,
        help="Source-plane FoV in arcsec (default 1.0). Setting this below "
        "the inner caustic radius (~0.2-0.3 arcsec for SIE+shear at "
        "theta_E=1) puts most source pixels in the multiply-imaged region, "
        "boosting per-pixel info. Source pixel scale = source-fov / source-res.",
    )
    parser.add_argument(
        "--source-amp",
        type=float,
        default=200.0,
        help="Linear multiplier on the z source surface brightness. With "
        "image pixel area 0.0025 sq arcsec, default 200 yields image-plane "
        "fluxes ~0.5 per pixel inside the lensed ring.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument(
        "--gp-lengthscale",
        type=float,
        required=True,
        help="RBF kernel length scale in source pixels for the GP source prior.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    n = load_scale(args.smoke)["sl_n"]

    rng = np.random.default_rng(args.seed)
    dim_z = args.source_res * args.source_res
    L = _build_gp_chol(args.source_res, args.gp_lengthscale)
    eta = rng.standard_normal(size=(n, dim_z)).astype(np.float32)
    z = eta @ L.T.astype(np.float32)

    n_lens = LENS_PARAM_LO.shape[0]
    lens_uniform = rng.uniform(size=(n, n_lens)).astype(np.float32)
    lens_params = LENS_PARAM_LO + lens_uniform * (LENS_PARAM_HI - LENS_PARAM_LO)

    print(
        f"strong-lensing prep: n={n}, source_res={args.source_res} (d={dim_z}), "
        f"image_res={args.image_res}, "
        f"source_amp={args.source_amp}, source_fov={args.source_fov}, n_lens_params={n_lens}",
        flush=True,
    )

    t0 = time.time()
    y_clean = _generate_clean_images(
        z,
        lens_params,
        args.image_res,
        args.source_res,
        args.source_amp,
        args.source_fov,
        args.chunk,
    )
    elapsed = time.time() - t0
    rate = n / elapsed
    print(
        f"clean images: shape={y_clean.shape}, "
        f"min={y_clean.min():.3e}, mean={y_clean.mean():.3e}, max={y_clean.max():.3e}",
        flush=True,
    )
    print(f"forward time: {elapsed:.1f}s ({rate:.1f} forwards/s)", flush=True)

    payload = {
        "z": torch.from_numpy(z),
        "y_clean": torch.from_numpy(y_clean),
        "lens_params": torch.from_numpy(lens_params),
        "lens_param_names": LENS_PARAM_NAMES,
        "lens_param_lo": torch.from_numpy(LENS_PARAM_LO),
        "lens_param_hi": torch.from_numpy(LENS_PARAM_HI),
        "source_resolution": args.source_res,
        "image_resolution": args.image_res,
        "source_amp": args.source_amp,
        "source_fov_arcsec": args.source_fov,
        "image_pixel_scale_arcsec": IMAGE_PIXEL_SCALE_ARCSEC,
        "psf_fwhm_arcsec": PSF_FWHM_ARCSEC,
        "lens_model_list": LENS_MODEL_LIST,
        "gp_lengthscale_pixels": args.gp_lengthscale,
        "gp_chol": torch.from_numpy(L.astype(np.float32)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
