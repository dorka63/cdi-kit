"""Tests for selectable Otsu/relative CDI shrink-wrap thresholding."""
import numpy as np
import pytest
from scipy.ndimage import binary_closing, gaussian_filter

from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.custom.cdi_common import CDIGeometry, cdi_support
from ptypy.custom.cdi_shrinkwrap import CDIShrinkWrap, shrinkwrap_support, _otsu_threshold
from ptypy.engines import by_name
from ptypy.simulations.cdi_simulation import make_complex_test_object
import ptypy.custom.cdi_hio

SHAPE = (64, 64)


def _run(engines):
    true_object, initial_support = make_complex_test_object(SHAPE)
    geometry = CDIGeometry.from_parameters(shape=SHAPE, energy_kev=8.0,
        distance_m=1.0, detector_psize_m=55e-6, ffttype="numpy")
    intensity = np.abs(geometry.forward(true_object)) ** 2
    p = u.Param(verbose_level="error", data_type="double")
    p.io = u.Param(rfile=None, autosave=u.Param(active=False),
        autoplot=u.Param(active=False), interaction=u.Param(active=False))
    p.scans = u.Param(cdi=u.Param(name="Vanilla", data=u.Param(
        name="CDIScan", intensity=intensity, energy=8.0, distance=1.0,
        psize=55e-6, support=u.Param(kind="array", array=initial_support))))
    p.engines = u.Param()
    for label, pars in engines.items():
        p.engines[label] = pars
    reconstruction = Ptycho(p, level=5)
    return reconstruction, next(iter(reconstruction.pods.values()))


def _object():
    obj = np.zeros((32, 32), dtype=complex)
    obj[12:20, 12:20] = 1.0 + 0.5j
    return obj


@pytest.mark.parametrize("method", ["otsu", "relative"])
def test_shrinkwrap_support_finds_bright_region(method):
    obj = _object()
    support = shrinkwrap_support(obj, sigma_px=1.0, threshold=0.2,
                                 threshold_method=method)
    assert support.dtype == bool
    assert support.shape == obj.shape
    assert support[16, 16]
    assert not support[0, 0]


@pytest.mark.parametrize(("sigma_px", "threshold"), [
    (0.0, 0.15), (-1.0, 0.15), (1.0, 0.0), (1.0, -0.1), (1.0, 1.0),
])
def test_shrinkwrap_support_rejects_invalid_parameters(sigma_px, threshold):
    with pytest.raises(ValueError):
        shrinkwrap_support(np.ones((8, 8), dtype=complex), sigma_px=sigma_px,
                           threshold=threshold, threshold_method="relative")


@pytest.mark.parametrize("method", ["otsu", "relative"])
def test_shrinkwrap_support_rejects_zero_object(method):
    with pytest.raises(ValueError, match="positive finite maximum"):
        shrinkwrap_support(np.zeros((8, 8), complex), sigma_px=1.0,
                           threshold_method=method)


def test_default_is_otsu_and_ignores_relative_threshold():
    obj = _object()
    expected = shrinkwrap_support(obj, sigma_px=1.0, threshold_method="otsu")
    np.testing.assert_array_equal(shrinkwrap_support(obj, sigma_px=1.0), expected)
    np.testing.assert_array_equal(
        shrinkwrap_support(obj, sigma_px=1.0, threshold=0.99), expected)
    relative = shrinkwrap_support(obj, sigma_px=1.0,
                                 threshold=0.99, threshold_method="relative")
    assert not np.array_equal(relative, expected)


def test_relative_preserves_original_rule():
    obj = _object()
    blurred = gaussian_filter(np.abs(obj), sigma=(1.0, 2.0))
    expected = blurred >= 0.2 * blurred.max()
    before = obj.copy()
    actual = shrinkwrap_support(obj, sigma_px=(1.0, 2.0),
                                threshold=0.2, threshold_method="relative")
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(obj, before)


def test_otsu_maximizes_between_class_variance():
    rng = np.random.default_rng(41)
    image = np.concatenate([rng.uniform(0.0, 0.1, 800), rng.uniform(0.7, 1.0, 224)])
    image = image.reshape(32, 32)
    normalized = image / image.max()
    counts, edges = np.histogram(normalized, bins=256,
                                range=(normalized.min(), normalized.max()))
    centers = (edges[:-1] + edges[1:]) / 2
    scores = []
    for split in range(255):
        left, right = counts[:split + 1], counts[split + 1:]
        if not left.sum() or not right.sum():
            scores.append(0.0)
        else:
            mean_left = np.sum(left * centers[:split + 1]) / left.sum()
            mean_right = np.sum(right * centers[split + 1:]) / right.sum()
            scores.append(left.sum() * right.sum() * (mean_left - mean_right)**2)
    cutoff = _otsu_threshold(image)
    selected = int(np.argmin(np.abs(centers[:-1] * image.max() - cutoff)))
    assert scores[selected] == pytest.approx(max(scores))
    assert cutoff == pytest.approx(centers[selected] * image.max())


def test_otsu_separates_two_levels():
    image = np.zeros((16, 16))
    image[4:12, 4:12] = 1.0
    cutoff = _otsu_threshold(image)
    assert 0.0 < cutoff < 1.0
    np.testing.assert_array_equal(image > cutoff, image.astype(bool))


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_otsu_support_is_scale_invariant(scale):
    obj = _object()
    expected = shrinkwrap_support(obj, sigma_px=1.0)
    actual = shrinkwrap_support(obj * scale, sigma_px=1.0)
    np.testing.assert_array_equal(actual, expected)


def test_positive_constant_has_full_support():
    support = shrinkwrap_support(np.ones((8, 8), complex), sigma_px=1.0)
    assert support.all()


def test_closing_is_applied_after_otsu():
    obj = _object()
    raw = shrinkwrap_support(obj, sigma_px=1.0)
    closed = shrinkwrap_support(obj, sigma_px=1.0, closing_iterations=1)
    np.testing.assert_array_equal(closed, binary_closing(raw, iterations=1))


def test_invalid_threshold_method():
    with pytest.raises(ValueError, match="threshold_method"):
        shrinkwrap_support(_object(), sigma_px=1.0, threshold_method="unknown")


@pytest.mark.parametrize("obj", [np.ones((4, 4, 4)), np.empty((0, 4)),
                                 np.full((4, 4), np.nan), np.full((4, 4), np.inf)])
def test_invalid_object(obj):
    with pytest.raises(ValueError):
        shrinkwrap_support(obj, sigma_px=1.0)


def test_cdi_shrinkwrap_is_registered():
    assert by_name("CDIShrinkWrap") is CDIShrinkWrap


@pytest.mark.parametrize("method", ["otsu", "relative"])
def test_cdi_shrinkwrap_updates_support(method):
    reconstruction, pod = _run({
        "e0": u.Param(name="CDIHIO", numiter=10),
        "e1": u.Param(name="CDIShrinkWrap", numiter=1, gaussian_sigma_px=1.0,
            minimum_sigma_px=1.0, threshold=0.2, threshold_method=method),
    })
    support = cdi_support(reconstruction, pod)
    assert support.shape == SHAPE
    assert support.dtype == bool
    assert support.any()
    assert len(reconstruction.runtime.iter_info) == 11
    assert reconstruction.engines["e1"].p.threshold_method == method


def test_cdi_shrinkwrap_default_is_otsu():
    reconstruction, _ = _run({"e0": u.Param(name="CDIShrinkWrap", numiter=1)})
    assert reconstruction.engines["e0"].p.threshold_method == "otsu"


def test_cdi_shrinkwrap_records_support_change():
    reconstruction, _ = _run({"e0": u.Param(name="CDIShrinkWrap", numiter=1)})
    info = reconstruction.runtime.iter_info[0]
    assert info["error"][0] == 0.0
    assert info["error"][1] == 0.0
    assert 0.0 <= info["error"][2] <= 1.0


def test_cdi_shrinkwrap_decays_sigma_to_configured_floor():
    reconstruction, _ = _run({"e0": u.Param(name="CDIShrinkWrap", numiter=3,
        gaussian_sigma_px=4.0, sigma_decay=0.5, minimum_sigma_px=1.5)})
    engine = next(iter(reconstruction.engines.values()))
    assert engine.update_count == 3
    assert engine.current_sigma_px == (1.5, 1.5)


def test_cdi_shrinkwrap_adds_its_citation():
    reconstruction, _ = _run({"e0": u.Param(name="CDIShrinkWrap", numiter=1)})
    comments = [entry["comment"] for entry in reconstruction.citations.entries]
    assert "The shrink-wrap support-update algorithm" in comments
