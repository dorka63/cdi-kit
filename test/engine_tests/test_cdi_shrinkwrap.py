import numpy as np
import pytest

from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.custom.cdi_common import CDIGeometry, cdi_support
from ptypy.custom.cdi_shrinkwrap import CDIShrinkWrap, shrinkwrap_support
from ptypy.engines import by_name
from ptypy.simulations.cdi_simulation import make_complex_test_object

import ptypy.custom.cdi_hio


SHAPE = (64, 64)


def _run(engines):
    true_object, initial_support = make_complex_test_object(SHAPE)
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
        support=u.Param(kind="array", array=initial_support),
    )

    p.engines = u.Param()
    for label, engine_pars in engines.items():
        p.engines[label] = engine_pars

    reconstruction = Ptycho(p, level=5)
    pod = next(iter(reconstruction.pods.values()))
    return reconstruction, pod


def test_shrinkwrap_support_finds_bright_region():
    obj = np.zeros((32, 32), dtype=complex)
    obj[12:20, 12:20] = 1.0 + 0.5j

    support = shrinkwrap_support(
        obj,
        sigma_px=1.0,
        threshold=0.2,
    )

    assert support.dtype == bool
    assert support.shape == obj.shape
    assert support[16, 16]
    assert not support[0, 0]


@pytest.mark.parametrize(
    ("sigma_px", "threshold"),
    [
        (0.0, 0.15),
        (-1.0, 0.15),
        (1.0, 0.0),
        (1.0, -0.1),
        (1.0, 1.0),
    ],
)
def test_shrinkwrap_support_rejects_invalid_parameters(
    sigma_px,
    threshold,
):
    with pytest.raises(ValueError):
        shrinkwrap_support(
            np.ones((8, 8), dtype=complex),
            sigma_px=sigma_px,
            threshold=threshold,
        )


def test_shrinkwrap_support_rejects_zero_object():
    with pytest.raises(ValueError, match="positive finite maximum"):
        shrinkwrap_support(
            np.zeros((8, 8), dtype=complex),
            sigma_px=1.0,
            threshold=0.15,
        )


def test_cdi_shrinkwrap_is_registered():
    assert by_name("CDIShrinkWrap") is CDIShrinkWrap


def test_cdi_shrinkwrap_updates_support():
    reconstruction, pod = _run({
        "e0": u.Param(name="CDIHIO", numiter=10),
        "e1": u.Param(
            name="CDIShrinkWrap",
            numiter=1,
            gaussian_sigma_px=1.0,
    	    minimum_sigma_px=1.0,
            threshold=0.2,
        ),
    })

    support = cdi_support(reconstruction, pod)

    assert support.shape == SHAPE
    assert support.dtype == bool
    assert support.any()
    assert len(reconstruction.runtime.iter_info) == 11


def test_cdi_shrinkwrap_records_support_change():
    reconstruction, _ = _run({
        "e0": u.Param(name="CDIShrinkWrap", numiter=1),
    })

    info = reconstruction.runtime.iter_info[0]

    assert info["error"][0] == 0.0
    assert info["error"][1] == 0.0
    assert 0.0 <= info["error"][2] <= 1.0


def test_cdi_shrinkwrap_decays_sigma_to_configured_floor():
    reconstruction, _ = _run({
        "e0": u.Param(
            name="CDIShrinkWrap",
            numiter=3,
            gaussian_sigma_px=4.0,
            sigma_decay=0.5,
            minimum_sigma_px=1.5,
        ),
    })

    engine = next(iter(reconstruction.engines.values()))

    assert engine.update_count == 3
    assert engine.current_sigma_px == (1.5, 1.5)


def test_cdi_shrinkwrap_adds_its_citation():
    reconstruction, _ = _run({
        "e0": u.Param(name="CDIShrinkWrap", numiter=1),
    })

    comments = [entry["comment"] for entry in reconstruction.citations.entries]

    assert "The shrink-wrap support-update algorithm" in comments