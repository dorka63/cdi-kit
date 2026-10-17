"""Tests for residual-free CDI ensembles and standard .ptyr saving."""
import numpy as np
import pytest
from ptypy import io, utils as u
from ptypy.core import Ptycho
from ptypy.custom.cdi_common import CDIGeometry, cdi_support
from ptypy.custom.cdi_ensemble import CDIEnsemble, svd_modes
from ptypy.simulations.cdi_simulation import make_complex_test_object


def parameters(propagation="farfield", filename=None):
    obj, support = make_complex_test_object((16, 16), support_radius_fraction=0.25)
    geo = CDIGeometry.from_parameters(shape=obj.shape, energy_kev=8.0,
        distance_m=1.0, detector_psize_m=55e-6, propagation=propagation, ffttype="numpy")
    intensity = np.abs(geo.forward(obj))**2
    mask = np.ones(obj.shape, dtype=bool)
    mask[7:9, 7:9] = False
    intensity[~mask] = 0
    p = u.Param(verbose_level="error", data_type="double")
    p.io = u.Param(rfile=None if filename is None else str(filename), rformat="minimal",
                   autosave=u.Param(active=False), autoplot=u.Param(active=False),
                   interaction=u.Param(active=False))
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


def test_seeds_are_reproducible_random_amplitudes():
    with CDIEnsemble(parameters(), seeds=(1, 2, 3)) as first, \
         CDIEnsemble(parameters(), seeds=(1, 2, 3)) as second:
        np.testing.assert_array_equal(first.fields, second.fields)
        assert not np.allclose(first.fields[0], first.fields[1])
        for index, seed in enumerate(first.seeds):
            expected = np.random.default_rng(seed).random(size=first.support.shape) * first.support
            np.testing.assert_array_equal(first.fields[index], expected)
        assert np.all(first.fields.imag == 0)
        assert first.result is None
        assert not hasattr(first, "final_errors")
        assert not hasattr(first, "selected_indices")


@pytest.mark.parametrize("propagation", ["farfield", "nearfield"])
def test_run_svd_support_and_continuation(propagation):
    with CDIEnsemble(parameters(propagation), seeds=(1, 2, 3)) as ensemble:
        p = ensemble.ptycho
        mask = ensemble.pod.mask.copy()
        data = ensemble.pod.diff.copy()
        assert ensemble.run() is ensemble
        assert ensemble.completed_stages == 1
        assert ensemble.iterations_per_member == 2
        assert np.isfinite(ensemble.fields).all()
        before = ensemble.fields.copy()
        expected_modes, expected_weights = svd_modes(before, num_modes=2)
        modes, weights = ensemble.svd(num_modes=2)
        assert modes.shape == (2, 16, 16)
        assert weights.shape == (3,)
        np.testing.assert_allclose(modes, expected_modes)
        np.testing.assert_allclose(weights, expected_weights)
        np.testing.assert_array_equal(ensemble.result, modes[0])
        np.testing.assert_array_equal(ensemble.fields, before)
        modes[0] = 0
        assert np.any(ensemble.result != 0)
        support = ensemble.support.copy()
        support[0, 0] = True
        ensemble.set_support(support)
        support[0, 0] = False
        assert ensemble.support[0, 0]
        np.testing.assert_array_equal(ensemble.fields, before)
        ensemble.run(u.Param(e00=u.Param(name="CDIRAAR", beta=0.9, numiter=2)))
        assert ensemble.result is None
        assert ensemble.ptycho is p
        assert ensemble.completed_stages == 2
        assert ensemble.iterations_per_member == 4
        np.testing.assert_array_equal(ensemble.pod.mask, mask)
        np.testing.assert_array_equal(ensemble.pod.diff, data)
        np.testing.assert_array_equal(cdi_support(p, ensemble.pod), ensemble.support)
        assert not p.engines


def test_repeated_stage_matches_long_er_chain():
    with CDIEnsemble(parameters(), seeds=(1, 2)) as short, \
         CDIEnsemble(parameters(), seeds=(1, 2)) as long:
        short.run().run()
        long.run(u.Param(e00=u.Param(name="CDIER", numiter=4)))
        np.testing.assert_allclose(short.fields, long.fields, atol=1e-10, rtol=1e-10)


def test_invalid_support_does_not_change_state():
    with CDIEnsemble(parameters(), seeds=(1, 2)) as ensemble:
        before = ensemble.support.copy()
        with pytest.raises(ValueError):
            ensemble.set_support(np.zeros((16, 16), dtype=bool))
        np.testing.assert_array_equal(ensemble.support, before)


def test_finalize_saves_one_primary_mode_automatically(tmp_path):
    path = tmp_path / "primary.ptyr"
    with CDIEnsemble(parameters(filename=path), seeds=(1, 2)) as ensemble:
        ensemble.run()
        assert not path.exists()
        modes, _ = ensemble.svd()
        np.testing.assert_array_equal(ensemble.result, modes[0])
        assert not path.exists()
        expected = ensemble.result.copy()
        fields = ensemble.fields.copy()
    assert path.is_file()
    content = io.h5read(str(path), "content")["content"]
    assert len(content["obj"]) == 1
    stored = next(iter(content["obj"].values()))["data"]
    assert stored.shape == (1, 16, 16)
    np.testing.assert_allclose(stored[0], expected, atol=1e-12)
    np.testing.assert_array_equal(ensemble.fields, fields)
    assert content["runtime"]["cdi_result"] == "svd_mode_0"
    assert len(content["runtime"]["iter_info"]) == 0
    assert "fields" not in content and "ensemble" not in content
    assert np.asarray(content["pars"]["scans"]["cdi"]["data"]["center"]).shape == (2,)
    loaded = Ptycho.load_run(str(path), load_data=False)
    loaded_data = next(iter(loaded.obj.storages.values())).data
    np.testing.assert_allclose(loaded_data[0], expected)
    ensemble.finalize()  # Idempotent.


def test_save_failure_restores_workspace(tmp_path):
    with CDIEnsemble(parameters(), seeds=(1, 2)) as ensemble:
        ensemble.run()
        ensemble.svd()
        new_support = ensemble.support.copy()
        new_support[0, 0] = True
        ensemble.set_support(new_support)
        previous = ensemble.pod.object.copy()
        previous_io = ensemble.ptycho.p.io
        previous_runtime = ensemble.ptycho.runtime
        path = tmp_path / "existing.ptyr"
        ensemble.save_run(path)
        with pytest.raises(RuntimeError):
            ensemble.save_run(path, force_overwrite=False)
        np.testing.assert_array_equal(ensemble.pod.object, previous)
        np.testing.assert_array_equal(ensemble.support, new_support)
        np.testing.assert_array_equal(cdi_support(ensemble.ptycho, ensemble.pod), new_support)
        assert ensemble.ptycho.p.io is previous_io
        assert ensemble.ptycho.runtime is previous_runtime


def test_exception_exit_does_not_save(tmp_path):
    path = tmp_path / "not_final.ptyr"
    ensemble = CDIEnsemble(parameters(filename=path), seeds=(1, 2))
    with pytest.raises(RuntimeError, match="abort"):
        with ensemble:
            ensemble.run()
            raise RuntimeError("abort")
    assert not path.exists()
    assert ensemble._closed
