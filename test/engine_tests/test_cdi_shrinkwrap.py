import numpy as np
import pytest

from ptypy.custom.cdi_shrinkwrap import ShrinkWrapSupport


def test_shrinkwrap_returns_boolean_support_with_input_shape():
    object_field = np.zeros((31, 35), dtype=np.complex128)
    object_field[10:21, 12:24] = 1.0 + 0.3j

    shrinkwrap = ShrinkWrapSupport(
        gaussian_sigma_px=1.0,
        threshold=0.2,
        sigma_decay=1.0,
        minimum_sigma_px=1.0,
    )

    support = shrinkwrap.update(object_field)

    assert support.dtype == bool
    assert support.shape == object_field.shape
    assert support.any()
    assert not support.all()
    assert support[15, 17]


def test_shrinkwrap_decay_stops_at_minimum_sigma():
    shrinkwrap = ShrinkWrapSupport(
        gaussian_sigma_px=3.0,
        threshold=0.2,
        sigma_decay=0.5,
        minimum_sigma_px=1.5,
    )

    object_field = np.ones((9, 9), dtype=np.complex128)

    shrinkwrap.update(object_field)
    assert shrinkwrap.current_sigma_px == pytest.approx((1.5, 1.5))

    shrinkwrap.update(object_field)
    assert shrinkwrap.current_sigma_px == pytest.approx((1.5, 1.5))
    assert shrinkwrap.update_count == 2


def test_shrinkwrap_accepts_anisotropic_sigmas():
    shrinkwrap = ShrinkWrapSupport(
        gaussian_sigma_px=(4.0, 2.0),
        minimum_sigma_px=(2.0, 1.0),
        sigma_decay=0.5,
    )

    object_field = np.ones((11, 13), dtype=np.complex128)

    shrinkwrap.update(object_field)

    assert shrinkwrap.current_sigma_px == pytest.approx((2.0, 1.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gaussian_sigma_px": 0.0},
        {"gaussian_sigma_px": (-1.0, 1.0)},
        {"minimum_sigma_px": 0.0},
        {"minimum_sigma_px": (1.0, -1.0)},
        {"threshold": 0.0},
        {"threshold": 1.0},
        {"sigma_decay": 0.0},
        {"sigma_decay": 1.1},
        {"closing_iterations": -1},
    ],
)
def test_shrinkwrap_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ShrinkWrapSupport(**kwargs)


def test_shrinkwrap_rejects_zero_object_field():
    shrinkwrap = ShrinkWrapSupport()

    with pytest.raises(ValueError, match="positive finite maximum"):
        shrinkwrap.update(np.zeros((12, 12), dtype=np.complex128))


def test_shrinkwrap_rejects_non_2d_object_field():
    shrinkwrap = ShrinkWrapSupport()

    with pytest.raises(ValueError, match="two-dimensional"):
        shrinkwrap.update(np.ones((3, 4, 5), dtype=np.complex128))