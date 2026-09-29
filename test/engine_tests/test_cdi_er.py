import numpy as np
import pytest

from ptypy.custom.cdi_common import (
    CDIGeometry,
    CDIProblem,
    rectangular_support,
)
from ptypy.custom.cdi_er import ErrorReduction
from ptypy.simulations.cdi_simulation import make_complex_test_object


@pytest.fixture
def er_problem_and_truth():
    shape = (64, 64)

    geometry = CDIGeometry.from_parameters(
        shape=shape,
        energy_kev=8.0,
        distance_m=1.0,
        detector_psize_m=55e-6,
        propagation="farfield",
        ffttype="numpy",
    )

    true_object, true_support = make_complex_test_object(shape)

    intensity = np.abs(geometry.forward(true_object)) ** 2

    problem = CDIProblem(
        geometry=geometry,
        measured_intensity=intensity,
        valid_mask=np.ones(shape, dtype=bool),
        support=true_support,
    )

    return problem, true_object


def random_phase_initial_guess(
    shape,
    support,
    seed=42,
):
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, size=shape)

    return support.astype(np.complex128) * np.exp(1j * phase)


def test_er_step_enforces_zero_outside_support(er_problem_and_truth):
    problem, _ = er_problem_and_truth

    initial = np.ones(problem.geometry.shape, dtype=np.complex128)

    algorithm = ErrorReduction(
        problem=problem,
        object_field=initial,
    )

    algorithm.step()

    np.testing.assert_array_equal(
        algorithm.object_field[~problem.support],
        0.0,
    )


def test_er_records_one_error_and_one_object_change(er_problem_and_truth):
    problem, _ = er_problem_and_truth

    algorithm = ErrorReduction(
        problem=problem,
        object_field=np.ones(problem.geometry.shape, dtype=np.complex128),
    )

    returned_error = algorithm.step()

    assert len(algorithm.amplitude_errors) == 1
    assert len(algorithm.object_changes) == 1
    assert returned_error == algorithm.amplitude_errors[0]
    assert returned_error >= 0.0
    assert algorithm.object_changes[0] >= 0.0


def test_er_reduces_data_error_on_noise_free_synthetic_data(
    er_problem_and_truth,
):
    problem, _ = er_problem_and_truth

    initial = random_phase_initial_guess(
        shape=problem.geometry.shape,
        support=problem.support,
        seed=4,
    )

    algorithm = ErrorReduction(
        problem=problem,
        object_field=initial,
    )

    initial_candidate, initial_error = problem.data_projection(
        algorithm.object_field
    )

    assert initial_candidate.shape == problem.geometry.shape
    assert initial_error > 0.0

    algorithm.run(80)

    final_detector_field = problem.geometry.forward(algorithm.object_field)
    final_error = problem.data_projection(
        algorithm.object_field
    )[1]

    assert np.isfinite(final_error)
    assert final_error < initial_error

    np.testing.assert_array_equal(
        algorithm.object_field[~problem.support],
        0.0,
    )

    assert np.all(np.isfinite(final_detector_field))


def test_er_rejects_nonpositive_iteration_count(er_problem_and_truth):
    problem, _ = er_problem_and_truth

    algorithm = ErrorReduction(
        problem=problem,
        object_field=np.ones(problem.geometry.shape, dtype=np.complex128),
    )

    with pytest.raises(ValueError, match="positive"):
        algorithm.run(0)


def test_er_accepts_real_initial_object(er_problem_and_truth):
    problem, _ = er_problem_and_truth

    algorithm = ErrorReduction(
        problem=problem,
        object_field=np.ones(problem.geometry.shape),
    )

    assert np.iscomplexobj(algorithm.object_field)