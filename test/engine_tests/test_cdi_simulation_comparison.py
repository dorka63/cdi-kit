"""Regression tests for pure ER/HIO/RAAR budgets and physical phantom size."""
from pathlib import Path
import importlib.util
from types import SimpleNamespace
import numpy as np
import pytest
from ptypy.simulations import cdi_simulation as sim

_path = Path(__file__).resolve().parents[2] / "tutorial/cdi/compare_er_hio.py"
_spec = importlib.util.spec_from_file_location("cdi_comparison_under_test", _path)
comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparison)


def test_hio_has_four_exact_ninety_iteration_intervals():
    assert comparison.build_strategy("HIO") == [
        ("CDIHIO",90,1.0),("CDIHIO",90,0.7),
        ("CDIHIO",90,0.4),("CDIHIO",90,0.1)]


@pytest.mark.parametrize("method", ["ER","HIO","RAAR"])
@pytest.mark.parametrize("total", [20,360,400])
def test_exact_pure_algorithm_budgets(method,total):
    blocks = comparison.build_strategy(method,total)
    assert sum(count for _,count,_ in blocks) == total
    assert all(name == {"ER":"CDIER","HIO":"CDIHIO","RAAR":"CDIRAAR"}[method]
               for name,_,_ in blocks)
    assert all(count > 0 for _,count,_ in blocks)


def test_raar_beta_rises_without_er():
    blocks = comparison.build_strategy("RAAR")
    beta = np.array([value for _,_,value in blocks])
    assert beta[0] == pytest.approx(0.75)
    assert beta[-1] == pytest.approx(0.99)
    assert np.all(np.diff(beta) >= 0)
    assert np.all((beta > 0) & (beta <= 1))


@pytest.mark.parametrize("total", [0,1,3,True,360.0])
def test_invalid_iteration_budgets(total):
    with pytest.raises(ValueError):comparison.build_strategy("HIO",total)


def test_hio_rejects_unequal_quarters():
    with pytest.raises(ValueError):comparison.build_strategy("HIO",361)


def _image(monkeypatch):
    image = np.zeros((32,32))
    image[10:22,6:26] = 1.
    monkeypatch.setattr(sim,"amplitude_from_image",lambda *args,**kwargs:image.copy())
    return image


def test_physical_object_uses_geometry_sampling_and_preserves_aspect(monkeypatch):
    _image(monkeypatch)
    geo = SimpleNamespace(shape=(128,128),object_psize_m=(1e-7,1e-7))
    field,mask = sim.make_physical_image_object(geo,"unused.png",diameter_m=4e-6,
                                               phase_model="zero")
    y,x = np.nonzero(mask)
    assert (y.max()-y.min()+1,x.max()-x.min()+1) == (24,40)
    assert field.shape == mask.shape == geo.shape
    assert np.all(field.imag == 0)
    assert np.all(mask == (np.abs(field)>0))


def test_physical_object_handles_anisotropic_sampling(monkeypatch):
    _image(monkeypatch)
    geo = SimpleNamespace(shape=(128,128),object_psize_m=(2e-7,1e-7))
    _,mask = sim.make_physical_image_object(geo,"unused.png",diameter_m=4e-6)
    y,x = np.nonzero(mask)
    assert (y.max()-y.min()+1,x.max()-x.min()+1) == (12,40)


@pytest.mark.parametrize("diameter", [0.,-1.,np.nan,np.inf])
def test_invalid_physical_diameter(monkeypatch,diameter):
    _image(monkeypatch)
    geo = SimpleNamespace(shape=(128,128),object_psize_m=(1e-7,1e-7))
    with pytest.raises(ValueError):sim.make_physical_image_object(geo,"unused.png",diameter_m=diameter)


def test_object_outside_grid_is_rejected(monkeypatch):
    _image(monkeypatch)
    geo = SimpleNamespace(shape=(16,16),object_psize_m=(1e-7,1e-7))
    with pytest.raises(ValueError):sim.make_physical_image_object(geo,"unused.png",diameter_m=10e-6)


def test_comparison_phase_alignment():
    obj = np.arange(1,17).reshape(4,4).astype(complex)
    assert comparison.complex_nrmse(obj,obj*np.exp(1.3j)) < 1e-14
