"""Serial/parallel equivalence tests for CDI ensembles."""
import numpy as np
import pytest
from ptypy import utils as u
from ptypy.custom.cdi_common import CDIGeometry
from ptypy.custom.cdi_ensemble import CDIEnsemble
from ptypy.simulations.cdi_simulation import make_complex_test_object


def _parameters(propagation):
    obj, support = make_complex_test_object((16, 16), support_radius_fraction=0.25)
    geo = CDIGeometry.from_parameters(shape=obj.shape, energy_kev=8., distance_m=1.,
        detector_psize_m=55e-6, propagation=propagation, ffttype="numpy")
    intensity = np.abs(geo.forward(obj))**2 + 0.1
    mask = np.ones(obj.shape, bool)
    mask[7:9, 7:9] = False
    p = u.Param(verbose_level="error", data_type="double")
    p.scans = u.Param(cdi=u.Param(name="Vanilla", propagation=propagation, ffttype="numpy",
        data=u.Param(name="CDIScan", intensity=intensity, mask=mask, energy=8.,
            distance=1., psize=55e-6, center="fftshift", autocenter=False,
            support=u.Param(kind="array", array=support),
            denoising=u.Param(active=True, name="sigma_clip", sigma=3., max_iter=5))))
    p.engines = u.Param(e00=u.Param(name="CDIHIO", beta=0.7, numiter=2),
                        e01=u.Param(name="CDIER", numiter=1))
    return p


@pytest.mark.parametrize("propagation", ["farfield", "nearfield"])
def test_parallel_matches_serial_across_support_updates(propagation):
    with CDIEnsemble(_parameters(propagation), seeds=(1, 2, 3), workers=1) as serial, \
         CDIEnsemble(_parameters(propagation), seeds=(1, 2, 3), workers=2) as parallel:
        np.testing.assert_array_equal(serial.fields, parallel.fields)
        serial.run()
        parallel.run()
        pool = parallel._pool
        np.testing.assert_allclose(parallel.fields, serial.fields, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(parallel.final_errors, serial.final_errors, rtol=1e-10, atol=1e-10)
        data = parallel.pod.diff.copy()
        mask = parallel.pod.mask.copy()
        support = serial.support.copy()
        support[0, 0] = True
        serial.set_support(support)
        parallel.set_support(support)
        serial.run()
        parallel.run()
        assert parallel._pool is pool
        assert parallel.completed_stages == 2
        np.testing.assert_allclose(parallel.fields, serial.fields, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(parallel.pod.diff, data)
        np.testing.assert_array_equal(parallel.pod.mask, mask)
        sm, sw = serial.svd(keep=2)
        pm, pw = parallel.svd(keep=2)
        np.testing.assert_allclose(pw, sw, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.abs(pm), np.abs(sm), rtol=1e-10, atol=1e-10)
    assert parallel._pool is None


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_count(workers):
    with pytest.raises(ValueError, match="workers"):
        CDIEnsemble(_parameters("farfield"), seeds=(1, 2), workers=workers)
