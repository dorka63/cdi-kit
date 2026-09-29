import numpy as np
import pytest

from ptypy.cdi.simulation import (
    circular_support,
    farfield_intensity,
    fft2_centered,
    ifft2_centered,
    make_complex_test_object,
)


def test_centered_fft_round_trip():
    rng = np.random.default_rng(42)
    field = rng.normal(size=(32, 48)) + 1j * rng.normal(size=(32, 48))

    recovered = ifft2_centered(fft2_centered(field))

    np.testing.assert_allclose(recovered, field, atol=1e-12, rtol=1e-12)


def test_farfield_intensity_is_real_and_nonnegative():
    obj, _ = make_complex_test_object((64, 64))

    intensity = farfield_intensity(obj)

    assert intensity.shape == obj.shape
    assert np.isrealobj(intensity)
    assert np.all(np.isfinite(intensity))
    assert np.all(intensity >= 0.0)


def test_circular_support_has_expected_center_pixel():
    support = circular_support((11, 11), radius_px=2.0)

    assert support.dtype == bool
    assert support[5, 5]
    assert not support[0, 0]


def test_circular_support_accepts_an_off_center_position():
    support = circular_support(
        shape=(20, 30),
        radius_px=3.0,
        center_px=(4.0, 22.0),
    )

    assert support[4, 22]
    assert not support[15, 5]


@pytest.mark.parametrize(
    "shape",
    [
        (0, 10),
        (10, 0),
        (10,),
        (10, 10, 10),
    ],
)
def test_circular_support_rejects_invalid_shape(shape):
    with pytest.raises(ValueError):
        circular_support(shape, radius_px=2.0)


def test_complex_phantom_is_zero_outside_support():
    obj, support = make_complex_test_object((64, 64))

    assert obj.shape == support.shape
    assert np.all(obj[~support] == 0.0)
    assert np.any(np.abs(obj[support]) > 0.0)