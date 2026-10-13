"""Tests for the custom CDI RAAR engine."""

import numpy as np
import pytest

from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.engines import by_name
from ptypy.custom.cdi_common import (
    CDIGeometry,
    cdi_support,
    masked_modulus_projection,
    rectangular_support,
)
from ptypy.custom.cdi_raar import CDIRAAR
from ptypy.simulations.cdi_simulation import make_complex_test_object


def _update(current, candidate, support, beta):
    """Exercise the numerical update without constructing Ptycho."""
    engine = object.__new__(CDIRAAR)
    engine.p = u.Param(beta=beta)

    return engine.object_update(current, candidate, support)


def _make_ptycho(propagation="farfield", beamstop=False):
    shape = (32, 32)

    obj, support = make_complex_test_object(
        shape=shape,
        support_radius_fraction=0.2,
    )

    geometry = CDIGeometry.from_parameters(
        shape=shape,
        energy_kev=8.0,
        distance_m=1.0,
        detector_psize_m=55e-6,
        propagation=propagation,
        ffttype="numpy",
    )

    intensity = np.abs(geometry.forward(obj)) ** 2
    mask = np.ones(shape, dtype=bool)

    if beamstop:
        mask[14:19, 14:19] = False
        intensity[~mask] = 0.0

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
    p.scans.cdi = u.Param(
        name="Vanilla",
        propagation=propagation,
        ffttype="numpy",
    )

    p.scans.cdi.data = u.Param(
        name="CDIScan",
        intensity=intensity,
        mask=mask,
        energy=8.0,
        distance=1.0,
        psize=55e-6,
        center="fftshift",
        autocenter=False,
        rebin=1,
        orientation=None,
        support=u.Param(
            kind="array",
            array=support,
        ),
    )

    p.engines = u.Param()

    reconstruction = Ptycho(p, level=2)
    pod = next(iter(reconstruction.pods.values()))

    return reconstruction, pod, mask


def test_raar_is_registered():
    assert by_name("CDIRAAR") is CDIRAAR


@pytest.mark.parametrize("beta", [0.0, 0.3, 0.5, 0.9, 1.0])
def test_update_matches_reflection_formula(beta):
    rng = np.random.default_rng(12)

    current = (
        rng.normal(size=(12, 16))
        + 1j * rng.normal(size=(12, 16))
    )
    candidate = (
        rng.normal(size=(12, 16))
        + 1j * rng.normal(size=(12, 16))
    )
    support = rectangular_support(
        current.shape,
        oversampling=2.0,
    )

    reflected_modulus = 2.0 * candidate - current
    reflected_support = (
        2.0 * np.where(support, reflected_modulus, 0.0)
        - reflected_modulus
    )

    expected = (
        0.5 * beta * (reflected_support + current)
        + (1.0 - beta) * candidate
    )

    result = _update(current, candidate, support, beta)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-13,
        atol=1e-13,
    )


def test_update_preserves_inputs():
    current = np.full((8, 8), 1.0 + 2.0j)
    candidate = np.full((8, 8), 3.0 - 1.0j)
    support = rectangular_support((8, 8), oversampling=2.0)

    current_before = current.copy()
    candidate_before = candidate.copy()
    support_before = support.copy()

    result = _update(current, candidate, support, beta=0.9)

    np.testing.assert_array_equal(current, current_before)
    np.testing.assert_array_equal(candidate, candidate_before)
    np.testing.assert_array_equal(support, support_before)

    assert not np.shares_memory(result, current)
    assert not np.shares_memory(result, candidate)


@pytest.mark.parametrize("beta", [0.0, 0.5, 0.9, 1.0])
def test_feasible_field_is_fixed_point(beta):
    support = rectangular_support((16, 16), oversampling=2.0)

    field = np.zeros((16, 16), dtype=complex)
    field[support] = 1.0 + 0.5j

    result = _update(
        current=field,
        candidate=field,
        support=support,
        beta=beta,
    )

    np.testing.assert_allclose(result, field, atol=1e-14)


def test_beta_zero_returns_modulus_projection():
    rng = np.random.default_rng(3)
    current = rng.normal(size=(8, 8)).astype(complex)
    candidate = rng.normal(size=(8, 8)).astype(complex)
    support = rectangular_support((8, 8), oversampling=2.0)

    result = _update(current, candidate, support, beta=0.0)

    np.testing.assert_array_equal(result, candidate)


@pytest.mark.parametrize(
    "beta",
    [-0.1, 1.1, np.nan, np.inf, -np.inf],
)
def test_invalid_beta_is_rejected(beta):
    reconstruction, _, _ = _make_ptycho()

    with pytest.raises(ValueError, match="beta"):
        CDIRAAR(
            reconstruction,
            u.Param(name="CDIRAAR", beta=beta, numiter=1),
        )


@pytest.mark.parametrize("propagation", ["farfield", "nearfield"])
@pytest.mark.parametrize("beamstop", [False, True])
def test_raar_runs_through_ptycho(propagation, beamstop):
    reconstruction, pod, mask = _make_ptycho(
        propagation=propagation,
        beamstop=beamstop,
    )

    mask_before = pod.mask.copy()
    intensity_before = pod.diff.copy()
    support_before = cdi_support(reconstruction, pod).copy()

    reconstruction.run(
        epars=u.Param(
            name="CDIRAAR",
            beta=0.9,
            numiter=5,
            numiter_contiguous=1,
        )
    )

    assert pod.object.shape == (32, 32)
    assert np.iscomplexobj(pod.object)
    assert np.isfinite(pod.object).all()

    np.testing.assert_array_equal(pod.mask, mask)
    np.testing.assert_array_equal(pod.mask, mask_before)
    np.testing.assert_array_equal(pod.diff, intensity_before)
    np.testing.assert_array_equal(
        cdi_support(reconstruction, pod),
        support_before,
    )

    history = reconstruction.runtime.iter_info
    assert len(history) == 5

    for info in history:
        error = np.asarray(info["error"])
        assert np.isfinite(error).all()
        assert error[0] >= 0.0
        assert error[2] >= 0.0


@pytest.mark.parametrize("propagation", ["farfield", "nearfield"])
def test_single_engine_iteration_matches_masked_raar(propagation):
    reconstruction, pod, _ = _make_ptycho(
        propagation=propagation,
        beamstop=True,
    )

    rng = np.random.default_rng(21)
    current = (
        rng.normal(size=pod.object.shape)
        + 1j * rng.normal(size=pod.object.shape)
    )

    pod.object = current.copy()

    # Ptycho.run normally creates this history before engine initialization.
    reconstruction.runtime.iter_info = []

    engine = CDIRAAR(
        reconstruction,
        u.Param(name="CDIRAAR", beta=0.8, numiter=1),
    )

    engine.initialize()
    engine.prepare()

    detector_field = pod.fw(pod.probe * current)

    projected = masked_modulus_projection(
        detector_field,
        np.sqrt(pod.diff),
        pod.mask,
    )

    np.testing.assert_array_equal(
        projected[~pod.mask],
        detector_field[~pod.mask],
    )

    candidate = pod.bw(projected)
    support = cdi_support(reconstruction, pod)

    reflected_modulus = 2.0 * candidate - current
    reflected_support = (
        2.0 * np.where(support, reflected_modulus, 0.0)
        - reflected_modulus
    )

    expected = (
        0.4 * (reflected_support + current)
        + 0.2 * candidate
    )

    # Compare the update before final global-phase normalization.
    engine.engine_iterate(1)

    np.testing.assert_allclose(
        pod.object,
        expected,
        rtol=1e-11,
        atol=1e-11,
    )

    assert engine.curiter == 1

def test_raar_registers_citation():
    reconstruction, _, _ = _make_ptycho()

    CDIRAAR(
        reconstruction,
        u.Param(name="CDIRAAR", beta=0.9, numiter=1),
    )

    comments = [
        entry["comment"]
        for entry in reconstruction.citations.entries
    ]

    assert "The RAAR phase-retrieval algorithm" in comments