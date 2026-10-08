import numpy as np
import pytest

from ptypy import utils as u
from ptypy.custom.cdi_common import (
    CDIScan,
    denoise_diffraction_frame,
    sigma_clip_background,
)


def test_disabled_returns_unchanged_copy():
    intensity = np.arange(30, dtype=np.float64).reshape(5, 6)
    result = denoise_diffraction_frame(
        intensity, None, {"active": False}
    )

    np.testing.assert_array_equal(result, intensity)
    assert not np.shares_memory(result, intensity)


def test_sigma_clip_uses_retained_mean():
    values = np.array([9, 10, 10, 11, 12, 100], dtype=float)
    expected = np.mean([9, 10, 10, 11, 12])

    assert sigma_clip_background(values) == pytest.approx(expected)


def test_sigma_clip_subtracts_background_and_clips():
    intensity = np.array(
        [[9, 10, 10], [11, 12, 100]], dtype=np.uint16
    )
    original = intensity.copy()
    valid = np.ones(intensity.shape, dtype=bool)

    result = denoise_diffraction_frame(
        intensity,
        valid,
        {"active": True, "name": "sigma_clip"},
    )

    expected = np.maximum(intensity.astype(float) - 10.4, 0)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(intensity, original)


def test_statistics_mask_excludes_signal():
    intensity = np.array(
        [[9, 10, 11], [100, 200, 300]], dtype=float
    )
    selected = np.zeros(intensity.shape, dtype=bool)
    selected[0] = True

    result = denoise_diffraction_frame(
        intensity,
        None,
        {
            "active": True,
            "name": "sigma_clip",
            "statistics_mask": selected,
        },
    )

    np.testing.assert_allclose(
        result, np.maximum(intensity - 10, 0)
    )


def test_invalid_pixels_are_excluded_and_mask_is_unchanged():
    intensity = np.array(
        [[9, 10, 11], [np.nan, 1000, 1000]], dtype=float
    )
    valid = np.array(
        [[True, True, True], [False, False, False]]
    )
    original_mask = valid.copy()

    result = denoise_diffraction_frame(
        intensity,
        valid,
        {"active": True, "name": "sigma_clip"},
    )

    np.testing.assert_array_equal(valid, original_mask)
    np.testing.assert_allclose(result[valid], [0, 0, 1])
    np.testing.assert_array_equal(result[~valid], 0)


def test_empty_statistics_selection_raises():
    intensity = np.ones((8, 8))

    with pytest.raises(ValueError, match="selects no valid pixels"):
        denoise_diffraction_frame(
            intensity,
            None,
            {
                "active": True,
                "name": "sigma_clip",
                "statistics_mask": np.zeros((8, 8), dtype=bool),
            },
        )


@pytest.mark.parametrize("mode", ["lowpass", "highpass"])
def test_butterworth_constant_frame(mode):
    intensity = np.full((17, 24), 10.0)

    result = denoise_diffraction_frame(
        intensity,
        None,
        {
            "active": True,
            "name": "butterworth",
            "mode": mode,
        },
    )

    expected = intensity if mode == "lowpass" else np.zeros_like(intensity)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_butterworth_reduces_checkerboard():
    y, x = np.indices((32, 48))
    intensity = 10.0 + (-1.0) ** (x + y)

    result = denoise_diffraction_frame(
        intensity,
        None,
        {
            "active": True,
            "name": "butterworth",
            "mode": "lowpass",
            "cutoff_frequency_ratio": 0.08,
            "npad": 0,
        },
    )

    assert result.std() < intensity.std() * 0.01
    assert result.shape == intensity.shape
    assert np.all(result >= 0)


def test_butterworth_missing_pixels_require_policy():
    intensity = np.ones((16, 16))
    valid = np.ones(intensity.shape, dtype=bool)
    valid[8, 8] = False

    with pytest.raises(ValueError, match="missing-pixel"):
        denoise_diffraction_frame(
            intensity,
            valid,
            {"active": True, "name": "butterworth"},
        )


def test_butterworth_nearest_preserves_mask():
    intensity = np.full((16, 16), 10.0)
    valid = np.ones(intensity.shape, dtype=bool)
    valid[6:10, 6:10] = False
    intensity[~valid] = np.nan
    original_mask = valid.copy()

    result = denoise_diffraction_frame(
        intensity,
        valid,
        {
            "active": True,
            "name": "butterworth",
            "mask_policy": "nearest",
        },
    )

    np.testing.assert_array_equal(valid, original_mask)
    np.testing.assert_allclose(result[valid], 10.0, atol=1e-12)
    np.testing.assert_array_equal(result[~valid], 0)


def test_cdiscan_correct_uses_common_detector_mask():
    intensity = np.array(
        [[9, 10, 11], [1000, 1000, 1000]], dtype=float
    )
    valid = np.array(
        [[True, True, True], [False, False, False]]
    )

    scan = CDIScan(
        u.Param(
            intensity=intensity,
            mask=valid,
            support=u.Param(
                kind="array",
                array=np.ones(intensity.shape, dtype=bool),
            ),
            denoising=u.Param(
                active=True,
                name="sigma_clip",
            ),
        )
    )
    scan.initialize()

    raw, _, weights = scan.load([0])
    original = raw[0].copy()
    data, returned_weights = scan.correct(raw, weights, scan.common)

    assert returned_weights is weights
    np.testing.assert_array_equal(raw[0], original)
    np.testing.assert_allclose(data[0][valid], [0, 0, 1])
    np.testing.assert_array_equal(scan.load_weight(), valid)