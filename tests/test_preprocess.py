import numpy as np
import pytest

import preprocess
import sample


def test_normalization_constant_matches_sample():
    """preprocess.A and sample.FLUX_A are deliberate duplicates; they must agree."""
    assert preprocess.A == sample.FLUX_A


def test_normalize_probes_range_and_round_trip():
    rng = np.random.default_rng(0)
    flux = rng.uniform(-1.0, 10.0, size=(64, 64)).astype(np.float32)
    x = preprocess.normalize_probes(flux, A=preprocess.A)
    assert x.dtype == np.float32
    assert x.min() >= -1.0 and x.max() <= 1.0
    recovered = preprocess.A * (x + 1.0) / 2.0
    np.testing.assert_allclose(recovered, np.clip(flux, 0.0, preprocess.A), atol=1e-5)


def test_center_crop_shape_and_centering():
    ar = np.arange(300 * 300, dtype=np.float32).reshape(300, 300)
    out = preprocess.center_crop(ar, 256)
    assert out.shape == (256, 256)
    assert out[128, 128] == ar[150, 150]


def test_center_crop_too_small_raises():
    with pytest.raises(ValueError):
        preprocess.center_crop(np.zeros((100, 100)), 256)


def test_check_for_corruption():
    ok = np.ones((10, 10), dtype=np.float32)
    assert not preprocess.check_for_corruption(ok)

    nanned = ok.copy()
    nanned[0, 0] = np.nan
    assert preprocess.check_for_corruption(nanned)

    zeroed = ok.copy()
    zeroed.flat[:40] = 0.0   # 40% zeros > 30% threshold
    assert preprocess.check_for_corruption(zeroed)
