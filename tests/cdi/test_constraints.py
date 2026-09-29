import numpy as np
import pytest

from ptypy.cdi.constraints import masked_modulus_projection


def test_projection_replaces_amplitude_at_valid_pixels():
    wavefield = np.array(
        [
            [3.0 + 4.0j, 1.0 + 0.0j],
            [2.0 + 2.0j, -4.0 + 3.0j],
        ]
    )
    measured_amplitude = np.array(
        [
            [10.0, 7.0],
            [5.0, 2.0],
        ]
    )
    valid_mask = np.array(
        [
            [True, False],
            [True, True],
        ]
    )

    result = masked_modulus_projection(
        wavefield,
        measured_amplitude,
        valid_mask,
    )

    np.testing.assert_allclose(
        np.abs(result[valid_mask]),
        measured_amplitude[valid_mask],
    )


def test_projection_keeps_phase_at_valid_nonzero_pixels():
    wavefield = np.array(
        [
            [3.0 + 4.0j, 1.0 + 0.0j],
            [2.0 + 2.0j, -4.0 + 3.0j],
        ]
    )
    measured_amplitude = np.full((2, 2), 7.0)
    valid_mask = np.ones((2, 2), dtype=bool)

    result = masked_modulus_projection(
        wavefield,
        measured_amplitude,
        valid_mask,
    )

    expected_phase = wavefield / np.abs(wavefield)

    np.testing.assert_allclose(
        result / np.abs(result),
        expected_phase,
    )


def test_projection_does_not_change_invalid_pixels():
    wavefield = np.array(
        [
            [3.0 + 4.0j, 1.0 - 2.0j],
            [2.0 + 2.0j, -4.0 + 3.0j],
        ]
    )
    measured_amplitude = np.full((2, 2), 7.0)
    valid_mask = np.array(
        [
            [True, False],
            [False, True],
        ]
    )

    result = masked_modulus_projection(
        wavefield,
        measured_amplitude,
        valid_mask,
    )

    np.testing.assert_array_equal(
        result[~valid_mask],
        wavefield[~valid_mask],
    )


def test_projection_uses_zero_phase_for_zero_wavefield():
    wavefield = np.array(
        [
            [0.0 + 0.0j, 1.0 + 0.0j],
            [2.0 + 0.0j, 3.0 + 0.0j],
        ]
    )
    measured_amplitude = np.full((2, 2), 5.0)
    valid_mask = np.ones((2, 2), dtype=bool)

    result = masked_modulus_projection(
        wavefield,
        measured_amplitude,
        valid_mask,
    )

    np.testing.assert_allclose(result[0, 0], 5.0 + 0.0j)


@pytest.mark.parametrize(
    ("wavefield", "amplitude", "mask"),
    [
        (
            np.ones((2, 2), dtype=np.complex128),
            np.ones((3, 3)),
            np.ones((2, 2), dtype=bool),
        ),
        (
            np.ones((2, 2), dtype=np.complex128),
            np.ones((2, 2)),
            np.ones((3, 3), dtype=bool),
        ),
    ],
)
def test_projection_rejects_shape_mismatch(wavefield, amplitude, mask):
    with pytest.raises(ValueError):
        masked_modulus_projection(wavefield, amplitude, mask)


def test_projection_rejects_negative_amplitude():
    with pytest.raises(ValueError, match="non-negative"):
        masked_modulus_projection(
            wavefield=np.ones((2, 2), dtype=np.complex128),
            measured_amplitude=np.array(
                [
                    [1.0, -1.0],
                    [1.0, 1.0],
                ]
            ),
            valid_mask=np.ones((2, 2), dtype=bool),
        )