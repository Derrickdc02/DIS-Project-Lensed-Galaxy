import math

import pytest
import torch

from lensing import build_lens_sim, shear_cartesian, SOURCE_PIXELSCALE
from sample import lens_forward


def test_shear_cartesian_known_angles():
    g1, g2 = shear_cartesian(0.05, 0.0)
    assert (g1, g2) == pytest.approx((0.05, 0.0))
    g1, g2 = shear_cartesian(0.05, math.pi / 4)
    assert (g1, g2) == pytest.approx((0.0, 0.05), abs=1e-12)


def test_shear_cartesian_round_trip():
    gamma, phi = 0.05, 0.4
    g1, g2 = shear_cartesian(gamma, phi)
    assert math.hypot(g1, g2) == pytest.approx(gamma)
    assert 0.5 * math.atan2(g2, g1) == pytest.approx(phi)


@pytest.fixture(scope="module")
def sim():
    return build_lens_sim(source_pixelscale=SOURCE_PIXELSCALE)


def test_lensed_image_shape(sim):
    x = torch.rand(256, 256)
    assert lens_forward(sim, x).shape == (256, 256)


def test_flat_sky_maps_to_sky(sim):
    """The +1/-1 shift contract: x = -1 (empty sky) everywhere must lens to -1
    everywhere, including out-of-FOV rays that caustics zero-pads."""
    x = -torch.ones(256, 256)
    y = lens_forward(sim, x)
    assert torch.allclose(y, -torch.ones_like(y), atol=1e-6)
