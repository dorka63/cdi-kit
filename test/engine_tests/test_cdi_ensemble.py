"""Tests for fixed-grid CDI ensembles."""
import numpy as np
import pytest
from ptypy import utils as u
from ptypy.custom.cdi_common import CDIGeometry, cdi_support, masked_amplitude_error
from ptypy.custom.cdi_ensemble import CDIEnsemble, svd_modes
from ptypy.simulations.cdi_simulation import make_complex_test_object


def parameters(propagation="farfield"):
    obj, support = make_complex_test_object((16, 16), support_radius_fraction=0.25)
    geo = CDIGeometry.from_parameters(shape=obj.shape, energy_kev=8.0,
        distance_m=1.0, detector_psize_m=55e-6, propagation=propagation, ffttype="numpy")
    intensity = np.abs(geo.forward(obj))**2
    mask = np.ones(obj.shape, dtype=bool)
    mask[7:9, 7:9] = False
    intensity[~mask] = 0
    p = u.Param(verbose_level="error", data_type="double")
    p.scans = u.Param(cdi=u.Param(name="Vanilla", propagation=propagation, ffttype="numpy",
        data=u.Param(name="CDIScan", intensity=intensity, mask=mask,
            energy=8.0, distance=1.0, psize=55e-6, center="fftshift", autocenter=False,
            support=u.Param(kind="array", array=support))))
    p.engines = u.Param(e00=u.Param(name="CDIER", numiter=2))
    return p


def test_svd_rank_one_phase_variants():
    obj = np.zeros((8, 8), dtype=complex)
    obj[2:6, 2:6] = 1 + 0.2j
    fields = np.stack([obj * np.exp(1j * phase) for phase in (0.1, 0.7, 1.3)])
    original = fields.copy()
    modes, weights = svd_modes(fields)
    expected = obj * np.exp(-1j * np.angle(obj.sum()))
    np.testing.assert_allclose(modes[0], expected, atol=1e-12)
    assert weights[0] == pytest.approx(1)
    np.testing.assert_array_equal(fields, original)


def test_svd_orthogonal_fields_are_not_spatially_aligned():
    fields = np.zeros((2, 8, 8), dtype=complex)
    fields[0, 1, 1] = 2
    fields[1, 6, 6] = 1
    modes, weights = svd_modes(fields, num_modes=2)
    np.testing.assert_allclose(weights, [0.8, 0.2], atol=1e-12)
    assert abs(modes[0, 1, 1]) == pytest.approx(np.sqrt(2))
    assert abs(modes[0, 6, 6]) < 1e-12


@pytest.mark.parametrize("fields", [np.zeros((2, 8, 8)), np.ones((8, 8)), np.full((2, 8, 8), np.nan)])
def test_svd_rejects_invalid_fields(fields):
    with pytest.raises(ValueError):
        svd_modes(fields)


def test_seeds_are_reproducible_and_distinct():
    first = CDIEnsemble(parameters(), seeds=(1, 2, 3))
    second = CDIEnsemble(parameters(), seeds=(1, 2, 3))
    np.testing.assert_array_equal(first.fields, second.fields)
    assert not np.allclose(first.fields[0], first.fields[1])


@pytest.mark.parametrize("propagation", ["farfield", "nearfield"])
def test_run_svd_support_and_continuation(propagation):
    ensemble = CDIEnsemble(parameters(propagation), seeds=(1, 2, 3))
    p = ensemble.ptycho
    mask = ensemble.pod.mask.copy()
    data = ensemble.pod.diff.copy()
    ensemble.run()
    assert ensemble.completed_stages == 1
    assert ensemble.iterations_per_member == 2
    assert np.isfinite(ensemble.fields).all()
    for index, field in enumerate(ensemble.fields):
        expected = masked_amplitude_error(ensemble.pod.fw(field), np.sqrt(data), mask)
        assert ensemble.final_errors[index] == pytest.approx(expected)
    before = ensemble.fields.copy()
    modes, weights = ensemble.svd(keep=2)
    assert modes.shape == (1, 16, 16)
    assert weights.sum() == pytest.approx(1)
    np.testing.assert_array_equal(ensemble.fields, before)
    support = ensemble.support.copy()
    support[0, 0] = True
    ensemble.set_support(support)
    support[0, 0] = False
    assert ensemble.support[0, 0]
    np.testing.assert_array_equal(ensemble.fields, before)
    ensemble.run(u.Param(e00=u.Param(name="CDIRAAR", beta=0.9, numiter=2)))
    assert ensemble.ptycho is p
    assert ensemble.completed_stages == 2
    assert ensemble.iterations_per_member == 4
    np.testing.assert_array_equal(ensemble.pod.mask, mask)
    np.testing.assert_array_equal(ensemble.pod.diff, data)
    np.testing.assert_array_equal(cdi_support(p, ensemble.pod), ensemble.support)
    assert not p.engines


def test_repeated_stage_matches_long_er_chain():
    short = CDIEnsemble(parameters(), seeds=(1, 2))
    long = CDIEnsemble(parameters(), seeds=(1, 2))
    short.run().run()
    long.run(u.Param(e00=u.Param(name="CDIER", numiter=4)))
    np.testing.assert_allclose(short.fields, long.fields, atol=1e-10, rtol=1e-10)


def test_invalid_support_does_not_change_state():
    ensemble = CDIEnsemble(parameters(), seeds=(1, 2))
    before = ensemble.support.copy()
    with pytest.raises(ValueError):
        ensemble.set_support(np.zeros((16, 16), dtype=bool))
    np.testing.assert_array_equal(ensemble.support, before)
