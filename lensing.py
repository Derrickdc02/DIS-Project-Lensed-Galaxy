"""Differentiable strong-lensing forward model: SIE + external shear.

Mirrors the lens model in the source-reconstruction paper (SIE plus external
shear). Both components share a single lens plane so their deflection angles
add. The source is a Pixelated grid, so a score-model image can be dropped in
directly via ``sim({'source': {'image': x}})`` and gradients flow back through
``sim`` for posterior sampling.

Built against caustics 1.6.
"""

import math

import caustics


def shear_cartesian(gamma: float, phi_gamma: float) -> tuple[float, float]:
    """Convert shear magnitude + angle (rad) to caustics cartesian components.

    Inverse: ``gamma = hypot(gamma_1, gamma_2)`` and
    ``phi_gamma = 0.5 * atan2(gamma_2, gamma_1)``.
    """
    return gamma * math.cos(2 * phi_gamma), gamma * math.sin(2 * phi_gamma)


def build_lens_sim(
    cosmology=None,
    *,
    image_size: int = 256,
    pixelscale: float = 0.04,   # arcsec/pixel (HST-like)
    z_l: float = 0.5,
    z_s: float = 1.5,
    # --- SIE (main lens galaxy) ---
    x0: float = 0.0,
    y0: float = 0.0,
    q: float = 0.7,             # axis ratio (minor/major)
    phi: float = math.pi / 6,   # position angle (rad)
    Rein: float = 1.2,          # Einstein radius (arcsec)
    # --- external shear (line-of-sight / environment) ---
    gamma_1: float = 0.03,
    gamma_2: float = 0.04,      # (0.03, 0.04) -> |gamma| = 0.05 at ~26.6 deg
    device=None,
):
    """Build a differentiable SIE + external-shear lensing simulator.

    Parameters
    ----------
    cosmology
        A caustics cosmology; defaults to ``FlatLambdaCDM`` if None.
    image_size, pixelscale
        Grid side length (pixels) and angular scale (arcsec/pixel). Field of
        view is ``image_size * pixelscale`` arcsec.
    z_l, z_s
        Lens and source redshifts (source must be behind the lens).
    x0, y0, q, phi, Rein
        SIE center, axis ratio, position angle (rad), and Einstein radius.
    gamma_1, gamma_2
        Cartesian external-shear components. Use :func:`shear_cartesian` to
        convert from magnitude + angle.
    device
        If given, the returned ``sim`` is moved to this device.

    Returns
    -------
    caustics.LensSource
        Callable forward operator: source image -> lensed image-plane image.
    """
    if cosmology is None:
        cosmology = caustics.FlatLambdaCDM(name="cosmo")

    sie = caustics.SIE(
        cosmology=cosmology, name="sie",
        z_l=z_l, z_s=z_s,
        x0=x0, y0=y0, q=q, phi=phi, Rein=Rein,
    )
    shear = caustics.ExternalShear(
        cosmology=cosmology, name="shear",
        z_l=z_l, z_s=z_s,
        x0=x0, y0=y0,
        gamma_1=gamma_1, gamma_2=gamma_2,
    )
    lens = caustics.SinglePlane(
        cosmology=cosmology, name="lens",
        z_l=z_l, z_s=z_s,
        lenses=[sie, shear],
    )
    source = caustics.Pixelated(
        name="source", x0=0.0, y0=0.0,
        pixelscale=pixelscale, shape=(image_size, image_size),
    )
    sim = caustics.LensSource(
        lens=lens, source=source,
        pixelscale=pixelscale, pixels_x=image_size, pixels_y=image_size,
        name="sim",
    )
    if device is not None:
        sim.to(device)
    return sim
