import numpy as np
import pytest
import torch

from sample import pixelate_image, to_display_flux, atomic_save, FLUX_A


def test_pixelate_2d_block_means():
    x = torch.arange(16.0).reshape(4, 4)
    expected = x.reshape(2, 2, 2, 2).permute(0, 2, 1, 3).reshape(2, 2, 4).mean(-1)
    assert torch.equal(pixelate_image(x, 2), expected)


def test_pixelate_3d_matches_per_sample_2d():
    x = torch.rand(3, 8, 8)
    out = pixelate_image(x, 2)
    assert out.shape == (3, 4, 4)
    for i in range(3):
        assert torch.equal(out[i], pixelate_image(x[i], 2))


def test_pixelate_factor_1_is_identity():
    x = torch.rand(8, 8)
    assert pixelate_image(x, 1) is x


def test_pixelate_bad_ndim_raises():
    with pytest.raises(ValueError):
        pixelate_image(torch.rand(1, 1, 8, 8), 2)


def test_to_display_flux_range_and_endpoints():
    x = torch.linspace(-1, 1, 11)
    flux = to_display_flux(x, floor=1e-3)
    assert flux.min() == pytest.approx(1e-3)   # x = -1 -> floored
    assert flux.max() == pytest.approx(FLUX_A)  # x = +1 -> A
    assert (flux > 0).all()


def test_to_display_flux_tensor_ndarray_agree():
    x = np.linspace(-1, 1, 11, dtype=np.float32)
    np.testing.assert_allclose(to_display_flux(torch.from_numpy(x)),
                               to_display_flux(x), rtol=1e-6)


def test_atomic_save_round_trip(tmp_path):
    path = tmp_path / "obj.pt"
    atomic_save({"a": torch.ones(3)}, path)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert torch.equal(torch.load(path, weights_only=False)["a"], torch.ones(3))
