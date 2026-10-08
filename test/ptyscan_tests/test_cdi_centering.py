import numpy as np
import pytest

from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.custom.cdi_common import (
    CDIScan,
    estimate_cdi_center,
)


def make_symmetric_pattern():
    shape = (65, 65)
    expected_center = np.array([30.27, 33.41])

    y, x = np.indices(shape, dtype=float)

    r2 = (
        (y - expected_center[0]) ** 2
        + (x - expected_center[1]) ** 2
    )

    intensity = (
        np.exp(-r2 / 80.0)
        + 0.4 * np.exp(-((np.sqrt(r2) - 17.0) / 2.0) ** 2)
    )

    mask = np.ones(shape, dtype=bool)
    mask[r2 < 36.0] = False
    mask[:, 10:12] = False

    return intensity, mask, expected_center


@pytest.mark.parametrize("method", ["inversion", "reflections"])
@pytest.mark.parametrize("use_mask", [False, True])
def test_fractional_center_estimation(method, use_mask):
    intensity, mask, expected_center = make_symmetric_pattern()

    if not use_mask:
        mask = np.ones(intensity.shape, dtype=bool)

    original_intensity = intensity.copy()
    original_mask = mask.copy()

    center, diagnostics = estimate_cdi_center(
        intensity,
        mask,
        method=method,
        initial_center=(32.0, 32.0),
        search_radius=4.0,
        tolerance=1e-4,
        min_overlap=0.5,
    )

    actual_center = np.asarray(center, dtype=float)

    assert np.linalg.norm(actual_center - expected_center) < 0.01
    assert diagnostics["method"] == method

    np.testing.assert_array_equal(
        intensity,
        original_intensity,
    )

    np.testing.assert_array_equal(
        mask,
        original_mask,
    )


def test_mass_center_estimation():
    intensity = np.zeros((9, 9), dtype=float)
    intensity[3, 4] = 3.0
    intensity[4, 4] = 1.0

    center, diagnostics = estimate_cdi_center(
        intensity,
        method="mass",
    )

    np.testing.assert_allclose(
        center,
        (3.25, 4.0),
        rtol=0,
        atol=1e-12,
    )

    assert diagnostics["method"] == "mass"


@pytest.mark.parametrize("method", ["inversion", "reflections"])
def test_constant_pattern_rejected(method):
    intensity = np.ones((20, 20), dtype=float)

    with pytest.raises(ValueError):
        estimate_cdi_center(
            intensity,
            method=method,
            search_radius=2.0,
        )


@pytest.mark.parametrize("method", ["inversion", "reflections"])
def test_cdiscan_integer_center_metadata(method):
    intensity, mask, expected_center = make_symmetric_pattern()
    expected_integer_center = np.rint(expected_center).astype(int)

    pars = u.Param(
        intensity=intensity,
        mask=mask,
        center=None,
        autocenter=True,
        autocenter_method=method,
        autocenter_initial_center=(32.0, 32.0),
        autocenter_search_radius=4.0,
        autocenter_tolerance=1e-4,
        autocenter_min_overlap=0.5,
        orientation=None,
        rebin=1,
        save=None,
        support=u.Param(
            kind="circle",
            radius=10.0,
        ),
    )

    scan = CDIScan(pars)
    scan.initialize()

    package = scan.auto(1)

    assert isinstance(package, dict)

    actual_center = np.asarray(
        package["common"]["center"],
        dtype=float,
    )

    np.testing.assert_array_equal(
        actual_center,
        expected_integer_center,
    )

    frame = package["iterable"][0]

    np.testing.assert_array_equal(
        frame["data"],
        intensity,
    )

    np.testing.assert_array_equal(
        frame["mask"],
        mask,
    )

    assert scan.centering_diagnostics


def test_cdiscan_mass_center_rounded():
    intensity = np.zeros((9, 9), dtype=float)
    intensity[3, 4] = 3.0
    intensity[4, 4] = 1.0

    pars = u.Param(
        intensity=intensity,
        center=None,
        autocenter=True,
        autocenter_method="mass",
        orientation=None,
        rebin=1,
        save=None,
        support=u.Param(
            kind="circle",
            radius=2.0,
        ),
    )

    scan = CDIScan(pars)
    scan.initialize()

    package = scan.auto(1)

    actual_center = np.asarray(
        package["common"]["center"],
        dtype=float,
    )

    np.testing.assert_array_equal(
        actual_center,
        (3, 4),
    )


def test_cdiscan_manual_center_preserved():
    intensity, mask, _ = make_symmetric_pattern()
    manual_center = (29, 35)

    pars = u.Param(
        intensity=intensity,
        mask=mask,
        center=manual_center,
        autocenter=False,
        autocenter_method="inversion",
        orientation=None,
        rebin=1,
        save=None,
        support=u.Param(
            kind="circle",
            radius=10.0,
        ),
    )

    scan = CDIScan(pars)
    scan.initialize()

    package = scan.auto(1)

    actual_center = np.asarray(
        package["common"]["center"],
        dtype=float,
    )

    np.testing.assert_array_equal(
        actual_center,
        manual_center,
    )

    np.testing.assert_array_equal(
        package["iterable"][0]["data"],
        intensity,
    )

    np.testing.assert_array_equal(
        package["iterable"][0]["mask"],
        mask,
    )


@pytest.mark.parametrize("method", ["inversion", "reflections"])
def test_cdi_integer_center_reaches_geometry(method):
    intensity, mask, expected_center = make_symmetric_pattern()
    expected_integer_center = np.rint(expected_center).astype(int)

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
        resample=1,
    )

    p.scans.cdi.data = u.Param(
        name="CDIScan",
        intensity=intensity,
        mask=mask,
        energy=8.0,
        distance=1.0,
        psize=1e-4,
        center=None,
        autocenter=True,
        autocenter_method=method,
        autocenter_initial_center=(32.0, 32.0),
        autocenter_search_radius=4.0,
        autocenter_tolerance=1e-4,
        autocenter_min_overlap=0.5,
        orientation=None,
        rebin=1,
        save=None,
        support=u.Param(
            kind="circle",
            radius=10.0,
        ),
    )

    p.engines = u.Param()

    reconstruction = Ptycho(p, level=2)
    model = reconstruction.model.scans["cdi"]

    metadata_center = np.asarray(
        model.ptyscan.meta.center,
        dtype=float,
    )

    geometry_center = np.asarray(
        model.geometries[0].p.center,
        dtype=float,
    )

    print(f"Method: {method}")
    print("Expected integer center:", expected_integer_center)
    print("Metadata center:", metadata_center)
    print("Geometry center:", geometry_center)

    np.testing.assert_array_equal(
        metadata_center,
        expected_integer_center,
    )

    np.testing.assert_array_equal(
        geometry_center,
        metadata_center,
    )