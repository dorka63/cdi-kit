import numpy as np
import pytest

from ptypy import utils as u
from ptypy.core import Ptycho

from ptypy.custom.cdi_common import (
    CDIGeometry,
    CDIProblem,
    circular_support,
    masked_amplitude_error,
    masked_modulus_projection,
    rectangular_support,
    support_from_array,
    support_from_autocorrelation,
    support_from_npy,
    remove_global_phase,
)
from ptypy.simulations.cdi_simulation import make_complex_test_object


STAR_PATH = "tutorial/cdi/data/star.png"


def _geometry(shape):
    return CDIGeometry.from_parameters(
        shape=shape,
        energy_kev=8.0,
        distance_m=1.0,
        detector_psize_m=55e-6,
        propagation="farfield",
        ffttype="numpy",
    )


def _small_problem(shape=(16, 16)):
    return CDIProblem(
        geometry=_geometry(shape),
        measured_intensity=np.ones(shape),
        valid_mask=np.ones(shape, dtype=bool),
        support=rectangular_support(shape=shape, oversampling=2.0),
    )


def _cdi_scan_params(intensity, mask=None):
    p = u.Param()
    p.verbose_level = "error"
    p.data_type = "double"
    p.io = u.Param(
        rfile=None,
        autosave=u.Param(active=False),
        autoplot=u.Param(active=False),
        interaction=u.Param(active=False),
    )
    p.scans = u.Param()
    p.scans.cdi = u.Param(name="Vanilla")
    p.scans.cdi.data = u.Param(
        name="CDIScan",
        intensity=intensity,
        mask=mask,
        energy=8.0,
        distance=1.0,
        psize=55e-6,
    )
    return p


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


@pytest.mark.parametrize("shape", [(0, 12), (12, 0), (12,), (12, 12, 12)])
def test_circular_support_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        circular_support(shape=shape, radius_px=2.0)


@pytest.mark.parametrize("radius_px", [0.0, -1.0])
def test_circular_support_rejects_nonpositive_radius(radius_px):
    with pytest.raises(ValueError, match="radius_px"):
        circular_support(shape=(16, 16), radius_px=radius_px)


def test_rectangular_support_from_scalar_oversampling():
    support = rectangular_support(shape=(120, 80), oversampling=4.0)

    assert support.dtype == bool
    assert support.shape == (120, 80)
    assert support.sum() == 30 * 20
    assert support[59, 39]
    assert support[60, 40]
    assert not support[0, 0]


def test_rectangular_support_from_anisotropic_oversampling():
    support = rectangular_support(shape=(120, 80), oversampling=(2.0, 4.0))

    assert support.sum() == 60 * 20


def test_rectangular_support_accepts_off_centre_position():
    support = rectangular_support(
        shape=(30, 40),
        oversampling=(3.0, 4.0),
        center_px=(7.0, 31.0),
    )

    assert support.sum() == 100
    assert support[7, 31]
    assert not support[25, 5]


@pytest.mark.parametrize(
    "oversampling",
    [0.0, -1.0, (0.0, 2.0), (2.0, 0.0), (0.5, 2.0), (2.0, 0.5), (2.0, 3.0, 4.0)],
)
def test_rectangular_support_rejects_invalid_oversampling(oversampling):
    with pytest.raises(ValueError):
        rectangular_support(shape=(32, 32), oversampling=oversampling)


def test_support_from_boolean_array_returns_independent_boolean_copy():
    external_support = np.array([[False, True], [True, False]], dtype=bool)

    support = support_from_array(external_support, shape=(2, 2))

    assert support.dtype == bool
    np.testing.assert_array_equal(support, external_support)

    support[0, 1] = False
    assert external_support[0, 1]


def test_support_from_real_array_uses_threshold():
    external_support = np.array([[0.0, 0.4, 0.6], [1.0, 0.5, 0.2]])

    support = support_from_array(external_support, shape=(2, 3), threshold=0.5)

    expected = np.array([[False, False, True], [True, False, False]])
    np.testing.assert_array_equal(support, expected)


def test_support_from_complex_array_uses_magnitude():
    external_support = np.array([[0.0 + 0.0j, 0.3 + 0.4j], [0.0 + 0.8j, 0.1 + 0.1j]])

    support = support_from_array(external_support, threshold=0.5)

    expected = np.array([[False, False], [True, False]])
    np.testing.assert_array_equal(support, expected)


def test_support_from_array_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        support_from_array(np.ones((4, 4), dtype=bool), shape=(5, 5))


def test_support_from_array_rejects_empty_support():
    with pytest.raises(ValueError, match="at least one True"):
        support_from_array(np.zeros((4, 4), dtype=bool))


def test_support_from_npy_loads_external_mask(tmp_path):
    expected = np.array([[False, True, False], [True, True, False]])

    support_path = tmp_path / "support.npy"
    np.save(support_path, expected)

    loaded = support_from_npy(str(support_path), shape=(2, 3))

    np.testing.assert_array_equal(loaded, expected)


def test_support_from_npy_rejects_non_npy_extension(tmp_path):
    with pytest.raises(ValueError, match=".npy"):
        support_from_npy(str(tmp_path / "support.txt"))


def test_masked_modulus_projection_replaces_only_valid_amplitudes():
    detector_field = np.array([[3.0 + 4.0j, 1.0 - 2.0j], [2.0 + 2.0j, -4.0 + 3.0j]])
    measured_amplitude = np.array([[10.0, 7.0], [5.0, 2.0]])
    valid_mask = np.array([[True, False], [True, True]])

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
    detector_field = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [0.0 + 1.0j, -1.0 + 1.0j]])

    projected = masked_modulus_projection(
        detector_field=detector_field,
        measured_amplitude=np.full((2, 2), 3.0),
        valid_mask=np.ones((2, 2), dtype=bool),
    )

    np.testing.assert_allclose(projected[0, 0], 3.0 + 0.0j)


def test_masked_amplitude_error_is_zero_for_matching_amplitude():
    detector_field = np.array([[3.0 + 4.0j, 1.0 + 0.0j], [0.0 + 2.0j, -1.0 - 1.0j]])

    error = masked_amplitude_error(
        detector_field,
        np.abs(detector_field),
        np.ones((2, 2), dtype=bool),
    )

    assert error == pytest.approx(0.0, abs=1e-14)


def test_geometry_forward_backward_round_trip():
    geometry = _geometry((32, 48))

    rng = np.random.default_rng(123)
    object_field = rng.normal(size=(32, 48)) + 1j * rng.normal(size=(32, 48))

    recovered = geometry.backward(geometry.forward(object_field))

    np.testing.assert_allclose(recovered, object_field, rtol=1e-12, atol=1e-12)


def test_cdi_problem_builds_amplitude_and_applies_data_projection():
    geometry = _geometry((24, 24))
    support = rectangular_support(shape=(24, 24), oversampling=3.0)
    intensity = np.abs(geometry.forward(support.astype(np.complex128))) ** 2

    problem = CDIProblem(
        geometry=geometry,
        measured_intensity=intensity,
        valid_mask=np.ones((24, 24), dtype=bool),
        support=support,
    )

    candidate, error = problem.data_projection(np.ones((24, 24), dtype=np.complex128))

    assert candidate.shape == (24, 24)
    assert np.iscomplexobj(candidate)
    assert error >= 0.0
    np.testing.assert_allclose(problem.measured_amplitude, np.sqrt(intensity))


def test_autocorrelation_support_pipeline_runs_for_siemens_star():
    object_field, _ = make_complex_test_object(
        shape=(512, 512),
        amplitude_model="image",
        amplitude_image_path=STAR_PATH,
        support_radius_fraction=0.25,
    )
    intensity = np.abs(_geometry(object_field.shape).forward(object_field)) ** 2

    autocorrelation_support = support_from_autocorrelation(
        intensity,
        gaussian_sigma_px=1.0,
        threshold=0.10,
        closing_iterations=2,
    )

    assert autocorrelation_support.dtype == bool
    assert autocorrelation_support.shape == object_field.shape
    assert autocorrelation_support.any()
    assert not autocorrelation_support.all()


def test_set_support_replaces_support_with_independent_copy():
    problem = _small_problem()

    new_support = np.zeros((16, 16), dtype=bool)
    new_support[4:8, 4:8] = True

    problem.set_support(new_support)

    np.testing.assert_array_equal(problem.support, new_support)
    assert problem.support.dtype == bool

    new_support[5, 5] = False
    assert problem.support[5, 5]


def test_set_support_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        _small_problem().set_support(np.ones((8, 8), dtype=bool))


def test_set_support_rejects_empty_support():
    with pytest.raises(ValueError, match="at least one True"):
        _small_problem().set_support(np.zeros((16, 16), dtype=bool))


def test_cdi_scan_loads_single_pattern_into_ptycho():
    intensity = np.random.default_rng(0).uniform(0.0, 10.0, size=(64, 64))

    P = Ptycho(_cdi_scan_params(intensity), level=2)

    diff_storages = list(P.diff.storages.values())
    assert len(diff_storages) == 1
    assert diff_storages[0].data.shape == (1, 64, 64)
    np.testing.assert_allclose(diff_storages[0].data[0], intensity)
    assert len(P.pods) == 1


def test_cdi_scan_passes_detector_mask():
    intensity = np.ones((32, 32))
    mask = np.ones((32, 32), dtype=bool)
    mask[:4, :] = False

    P = Ptycho(_cdi_scan_params(intensity, mask), level=2)

    mask_storage = list(P.mask.storages.values())[0]
    np.testing.assert_array_equal(mask_storage.data[0], mask)

def test_remove_global_phase_sets_support_sum_phase_to_zero():
    obj = np.array(
        [
            [0.0 + 0.0j, 1.0 + 1.0j],
            [2.0 + 2.0j, 1.0 + 1.0j],
        ]
    )
    support = np.ones((2, 2), dtype=bool)

    cleaned = remove_global_phase(obj, support)

    assert np.angle(cleaned[support].sum()) == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_remove_global_phase_preserves_far_field_intensity():
    geometry = _geometry((32, 32))
    rng = np.random.default_rng(11)

    obj = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    support = np.ones((32, 32), dtype=bool)

    cleaned = remove_global_phase(obj, support)

    np.testing.assert_allclose(
        np.abs(geometry.forward(cleaned)) ** 2,
        np.abs(geometry.forward(obj)) ** 2,
        rtol=1e-12,
        atol=1e-12,
    )
