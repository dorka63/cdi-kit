import numpy as np
import pytest

from ptypy.custom.cdi_common import (
    CDIGeometry,
    CDIProblem,
    circular_support,
    masked_amplitude_error,
    masked_modulus_projection,
    rectangular_support,
    support_from_array,
    support_from_npy,
)


def test_circular_support_contains_centre_and_excludes_corner():
    support = circular_support(shape=(11, 11), radius_px=2.0)

    assert support.dtype == bool
    assert support.shape == (11, 11)
    assert support[5, 5]
    assert not support[0, 0]


def test_circular_support_accepts_off_centre_position():
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
        (0, 12),
        (12, 0),
        (12,),
        (12, 12, 12),
    ],
)
def test_circular_support_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        circular_support(shape=shape, radius_px=2.0)


@pytest.mark.parametrize("radius_px", [0.0, -1.0])
def test_circular_support_rejects_nonpositive_radius(radius_px):
    with pytest.raises(ValueError, match="radius_px"):
        circular_support(shape=(16, 16), radius_px=radius_px)


def test_rectangular_support_from_scalar_oversampling():
    support = rectangular_support(
        shape=(120, 80),
        oversampling=4.0,
    )

    assert support.dtype == bool
    assert support.shape == (120, 80)

    # 120 / 4 = 30 rows, 80 / 4 = 20 columns.
    assert support.sum() == 30 * 20
    assert support[59, 39]
    assert support[60, 40]
    assert not support[0, 0]


def test_rectangular_support_from_anisotropic_oversampling():
    support = rectangular_support(
        shape=(120, 80),
        oversampling=(2.0, 4.0),
    )

    # 120 / 2 = 60 rows, 80 / 4 = 20 columns.
    assert support.sum() == 60 * 20


def test_rectangular_support_accepts_off_centre_position():
    support = rectangular_support(
        shape=(30, 40),
        oversampling=(3.0, 4.0),
        center_px=(7.0, 31.0),
    )

    # Rectangle dimensions: 10 x 10.
    assert support.sum() == 100
    assert support[7, 31]
    assert not support[25, 5]


@pytest.mark.parametrize(
    "oversampling",
    [
        0.0,
        -1.0,
        (0.0, 2.0),
        (2.0, 0.0),
        (0.5, 2.0),
        (2.0, 0.5),
        (2.0, 3.0, 4.0),
    ],
)
def test_rectangular_support_rejects_invalid_oversampling(oversampling):
    with pytest.raises(ValueError):
        rectangular_support(shape=(32, 32), oversampling=oversampling)


def test_support_from_boolean_array_returns_independent_boolean_copy():
    external_support = np.array(
        [
            [False, True],
            [True, False],
        ],
        dtype=bool,
    )

    support = support_from_array(external_support, shape=(2, 2))

    assert support.dtype == bool
    np.testing.assert_array_equal(support, external_support)

    support[0, 1] = False
    assert external_support[0, 1]


def test_support_from_real_array_uses_threshold():
    external_support = np.array(
        [
            [0.0, 0.4, 0.6],
            [1.0, 0.5, 0.2],
        ],
        dtype=float,
    )

    support = support_from_array(
        external_support,
        shape=(2, 3),
        threshold=0.5,
    )

    expected = np.array(
        [
            [False, False, True],
            [True, False, False],
        ],
        dtype=bool,
    )

    np.testing.assert_array_equal(support, expected)


def test_support_from_complex_array_uses_magnitude():
    external_support = np.array(
        [
            [0.0 + 0.0j, 0.3 + 0.4j],
            [0.0 + 0.8j, 0.1 + 0.1j],
        ]
    )

    support = support_from_array(
        external_support,
        threshold=0.5,
    )

    expected = np.array(
        [
            [False, False],
            [True, False],
        ],
        dtype=bool,
    )

    np.testing.assert_array_equal(support, expected)


def test_support_from_array_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        support_from_array(
            np.ones((4, 4), dtype=bool),
            shape=(5, 5),
        )


def test_support_from_array_rejects_empty_support():
    with pytest.raises(ValueError, match="at least one True"):
        support_from_array(
            np.zeros((4, 4), dtype=bool),
        )


def test_support_from_npy_loads_external_mask(tmp_path):
    expected = np.array(
        [
            [False, True, False],
            [True, True, False],
        ],
        dtype=bool,
    )

    support_path = tmp_path / "support.npy"
    np.save(support_path, expected)

    loaded = support_from_npy(
        str(support_path),
        shape=(2, 3),
    )

    np.testing.assert_array_equal(loaded, expected)


def test_support_from_npy_rejects_non_npy_extension(tmp_path):
    invalid_path = tmp_path / "support.txt"

    with pytest.raises(ValueError, match=".npy"):
        support_from_npy(str(invalid_path))


def test_masked_modulus_projection_replaces_only_valid_amplitudes():
    detector_field = np.array(
        [
            [3.0 + 4.0j, 1.0 - 2.0j],
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
        ],
        dtype=bool,
    )

    projected = masked_modulus_projection(
        detector_field=detector_field,
        measured_amplitude=measured_amplitude,
        valid_mask=valid_mask,
    )

    np.testing.assert_allclose(
        np.abs(projected[valid_mask]),
        measured_amplitude[valid_mask],
    )

    np.testing.assert_array_equal(
        projected[~valid_mask],
        detector_field[~valid_mask],
    )


def test_masked_modulus_projection_has_deterministic_zero_phase():
    detector_field = np.array(
        [
            [0.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 1.0j, -1.0 + 1.0j],
        ]
    )

    measured_amplitude = np.full((2, 2), 3.0)
    valid_mask = np.ones((2, 2), dtype=bool)

    projected = masked_modulus_projection(
        detector_field=detector_field,
        measured_amplitude=measured_amplitude,
        valid_mask=valid_mask,
    )

    np.testing.assert_allclose(projected[0, 0], 3.0 + 0.0j)


def test_masked_amplitude_error_is_zero_for_matching_amplitude():
    detector_field = np.array(
        [
            [3.0 + 4.0j, 1.0 + 0.0j],
            [0.0 + 2.0j, -1.0 - 1.0j],
        ]
    )

    measured_amplitude = np.abs(detector_field)
    valid_mask = np.ones((2, 2), dtype=bool)

    error = masked_amplitude_error(
        detector_field,
        measured_amplitude,
        valid_mask,
    )

    assert error == pytest.approx(0.0, abs=1e-14)


def test_geometry_forward_backward_round_trip():
    geometry = CDIGeometry.from_parameters(
        shape=(32, 48),
        energy_kev=8.0,
        distance_m=1.0,
        detector_psize_m=55e-6,
        propagation="farfield",
        ffttype="numpy",
    )

    rng = np.random.default_rng(123)
    object_field = (
        rng.normal(size=(32, 48))
        + 1j * rng.normal(size=(32, 48))
    ).astype(np.complex128)

    recovered = geometry.backward(geometry.forward(object_field))

    np.testing.assert_allclose(
        recovered,
        object_field,
        rtol=1e-12,
        atol=1e-12,
    )


def test_cdi_problem_builds_amplitude_and_applies_data_projection():
    geometry = CDIGeometry.from_parameters(
        shape=(24, 24),
        energy_kev=8.0,
        distance_m=1.0,
        detector_psize_m=55e-6,
        propagation="farfield",
        ffttype="numpy",
    )

    support = rectangular_support(
        shape=(24, 24),
        oversampling=3.0,
    )

    true_object = support.astype(np.complex128)
    intensity = np.abs(geometry.forward(true_object)) ** 2

    problem = CDIProblem(
        geometry=geometry,
        measured_intensity=intensity,
        valid_mask=np.ones((24, 24), dtype=bool),
        support=support,
    )

    initial_object = np.ones((24, 24), dtype=np.complex128)
    candidate, error = problem.data_projection(initial_object)

    assert candidate.shape == (24, 24)
    assert np.iscomplexobj(candidate)
    assert error >= 0.0

    np.testing.assert_allclose(
        problem.measured_amplitude,
        np.sqrt(intensity),
    )