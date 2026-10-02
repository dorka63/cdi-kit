import numpy as np

from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.custom.cdi_common import CDIGeometry
from ptypy.custom.cdi_hio import CDIHIO
import ptypy.custom.cdi_er
from ptypy.engines import by_name
from ptypy.simulations.cdi_simulation import make_complex_test_object


SHAPE = (64, 64)


def _run(engines):
    true_object, support = make_complex_test_object(SHAPE)
    geometry = CDIGeometry.from_parameters(
        shape=SHAPE,
        energy_kev=8.0,
        distance_m=1.0,
        detector_psize_m=55e-6,
        ffttype="numpy",
    )
    intensity = np.abs(geometry.forward(true_object)) ** 2

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
        energy=8.0,
        distance=1.0,
        psize=55e-6,
        support=u.Param(kind="array", array=support),
    )
    p.engines = u.Param()
    for label, engine_pars in engines.items():
        p.engines[label] = engine_pars

    P = Ptycho(p, level=5)
    return P, list(P.pods.values())[0].object, support


def test_cdihio_is_registered():
    assert by_name("CDIHIO") is CDIHIO


def test_cdihio_uses_beta_parameter():
    P, _, _ = _run({"e0": u.Param(name="CDIHIO", numiter=1, beta=0.5)})

    assert list(P.engines.values())[0].p.beta == 0.5


def test_cdihio_reduces_amplitude_error_on_noise_free_data():
    P, obj, _ = _run({"e0": u.Param(name="CDIHIO", numiter=100)})

    errors = [info["error"][0] for info in P.runtime.iter_info]

    assert len(errors) == 100
    assert min(errors[-10:]) < errors[0]
    assert np.all(np.isfinite(obj))


def test_cdihio_adds_its_citation():
    P, _, _ = _run({"e0": u.Param(name="CDIHIO", numiter=1)})

    comments = [entry["comment"] for entry in P.citations.entries]

    assert "The Hybrid Input-Output phase-retrieval algorithm" in comments


def test_hio_then_er_chain_shares_object():
    P, obj, support = _run({
        "e0": u.Param(name="CDIHIO", numiter=20),
        "e1": u.Param(name="CDIER", numiter=5),
    })

    assert len(P.runtime.iter_info) == 25
    np.testing.assert_array_equal(obj[~support], 0.0)