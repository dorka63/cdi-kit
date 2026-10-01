import numpy as np

from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.custom.cdi_common import CDIGeometry
from ptypy.custom.cdi_er import CDIER
from ptypy.engines import by_name
from ptypy.simulations.cdi_simulation import make_complex_test_object


SHAPE = (64, 64)


def _simulated_data():
    true_object, support = make_complex_test_object(SHAPE)

    geometry = CDIGeometry.from_parameters(
        shape=SHAPE,
        energy_kev=8.0,
        distance_m=1.0,
        detector_psize_m=55e-6,
        ffttype="numpy",
    )
    intensity = np.abs(geometry.forward(true_object)) ** 2

    return intensity, support


def _run_cdier(numiter):
    intensity, support = _simulated_data()

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
    p.engines.e0 = u.Param(name="CDIER", numiter=numiter)

    P = Ptycho(p, level=5)
    pod = list(P.pods.values())[0]

    return P, pod.object, support


def test_cdier_is_registered():
    assert by_name("CDIER") is CDIER


def test_cdier_enforces_zero_outside_support():
    _, obj, support = _run_cdier(numiter=5)

    np.testing.assert_array_equal(obj[~support], 0.0)


def test_cdier_records_one_error_entry_per_iteration():
    P, _, _ = _run_cdier(numiter=5)

    errors = np.array([info["error"] for info in P.runtime.iter_info])

    assert errors.shape == (5, 3)
    assert np.all(np.isfinite(errors))
    assert np.all(errors >= 0.0)


def test_cdier_reduces_amplitude_error_on_noise_free_data():
    P, obj, _ = _run_cdier(numiter=80)

    amplitude_errors = [info["error"][0] for info in P.runtime.iter_info]

    assert amplitude_errors[-1] < amplitude_errors[0]
    assert np.all(np.isfinite(obj))


def test_cdier_adds_its_citation():
    P, _, _ = _run_cdier(numiter=1)

    comments = [entry["comment"] for entry in P.citations.entries]

    assert "The Error Reduction phase-retrieval algorithm" in comments