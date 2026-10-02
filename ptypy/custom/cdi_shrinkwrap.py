"""Shrink-wrap support-update engine for custom CDI reconstructions."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter

from ptypy.core.manager import Full, Vanilla
from ptypy.custom.cdi_common import cdi_support, set_cdi_support
from ptypy.engines import register
from ptypy.engines.base import BaseEngine


Array = np.ndarray


SHRINKWRAP_ARTICLE = dict(
    comment="The shrink-wrap support-update algorithm",
    title="X-ray image reconstruction from a diffraction pattern alone",
    author="Marchesini S. et al.",
    journal="Physical Review B",
    volume=68,
    year=2003,
    page="140101(R)",
    doi="10.1103/PhysRevB.68.140101",
)


def _validate_sigma(
    sigma: float | tuple[float, float] | Array,
    *,
    parameter_name: str,
) -> Array:
    """Normalize a scalar or pair of Gaussian widths to two positive floats."""
    sigma_array = np.asarray(sigma, dtype=float)

    if sigma_array.ndim > 1 or sigma_array.size not in {1, 2}:
        raise ValueError(
            f"{parameter_name} must be a scalar or a two-value tuple."
        )

    if sigma_array.size == 1:
        sigma_array = np.repeat(sigma_array, 2)

    if not np.all(np.isfinite(sigma_array)) or np.any(sigma_array <= 0.0):
        raise ValueError(
            f"{parameter_name} must contain positive finite values."
        )

    return sigma_array.astype(np.float64, copy=True)


def _validate_closing_iterations(closing_iterations: int) -> None:
    """Validate the optional binary-closing count."""
    if isinstance(closing_iterations, bool) or (
        not isinstance(closing_iterations, int) or closing_iterations < 0
    ):
        raise ValueError(
            "closing_iterations must be a non-negative integer."
        )


def shrinkwrap_support(
    object_field: Array,
    *,
    sigma_px: float | tuple[float, float] | Array,
    threshold: float,
    closing_iterations: int = 0,
) -> Array:
    """Compute a boolean shrink-wrap support from a complex object field."""
    sigma = _validate_sigma(sigma_px, parameter_name="sigma_px")
    _validate_closing_iterations(closing_iterations)

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between 0 and 1.")

    object_field = np.asarray(object_field)

    if object_field.ndim != 2:
        raise ValueError("object_field must be a two-dimensional array.")

    if not np.all(np.isfinite(object_field)):
        raise ValueError("object_field must contain finite values.")

    blurred_amplitude = gaussian_filter(
        np.abs(object_field),
        sigma=tuple(sigma),
    )
    maximum = float(blurred_amplitude.max())

    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(
            "blurred object amplitude has no positive finite maximum."
        )

    support = blurred_amplitude >= threshold * maximum

    if closing_iterations:
        support = binary_closing(
            support,
            iterations=closing_iterations,
        )

    support = np.asarray(support, dtype=bool)

    if not np.any(support):
        raise RuntimeError(
            "Shrink-wrap thresholding produced an empty support."
        )

    return support


@register()
class CDIShrinkWrap(BaseEngine):
    """
    Shrink-wrap support update for single-frame CDI.

    Defaults:

    [name]
    default = CDIShrinkWrap
    type = str
    help =

    [gaussian_sigma_px]
    default = 3.0
    type = float
    lowlim = 0.0
    help = Initial Gaussian blur width in pixels

    [threshold]
    default = 0.20
    type = float
    lowlim = 0.0
    uplim = 1.0
    help = Relative blurred-amplitude threshold

    [sigma_decay]
    default = 0.99
    type = float
    lowlim = 0.0
    uplim = 1.0
    help = Multiplicative Gaussian-width reduction after each update

    [minimum_sigma_px]
    default = 1.5
    type = float
    lowlim = 0.0
    help = Lower bound for Gaussian blur width in pixels

    [closing_iterations]
    default = 0
    type = int
    lowlim = 0
    help = Optional binary-closing iterations after thresholding

    """

    SUPPORTED_MODELS = [Vanilla, Full]

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)

        self._sigma = _validate_sigma(
            self.p.gaussian_sigma_px,
            parameter_name="gaussian_sigma_px",
        )
        self._minimum_sigma = _validate_sigma(
            self.p.minimum_sigma_px,
            parameter_name="minimum_sigma_px",
        )

        if np.any(self._minimum_sigma > self._sigma):
            raise ValueError(
                "minimum_sigma_px must not exceed gaussian_sigma_px."
            )

        if not 0.0 < self.p.threshold < 1.0:
            raise ValueError("threshold must lie strictly between 0 and 1.")

        if not 0.0 < self.p.sigma_decay <= 1.0:
            raise ValueError(
                "sigma_decay must lie in the interval (0, 1]."
            )

        _validate_closing_iterations(self.p.closing_iterations)

        self.update_count = 0
        ptycho_parent.citations.add_article(**SHRINKWRAP_ARTICLE)

    @property
    def current_sigma_px(self) -> tuple[float, float]:
        """Gaussian width that will be used by the next support update."""
        return (float(self._sigma[0]), float(self._sigma[1]))

    def engine_initialize(self):
        """Shrink-wrap has no separate initialization phase."""
        pass

    def engine_prepare(self):
        """Shrink-wrap has no per-block preparation phase."""
        pass

    def engine_iterate(self, num=1):
        error_dct = {}

        for _ in range(num):
            for name, diff_view in self.di.views.items():
                if not diff_view.active:
                    continue

                for pod in diff_view.pods.values():
                    old_support = cdi_support(self.ptycho, pod)

                    new_support = shrinkwrap_support(
                        pod.object,
                        sigma_px=self._sigma,
                        threshold=self.p.threshold,
                        closing_iterations=self.p.closing_iterations,
                    )
                    set_cdi_support(self.ptycho, pod, new_support)

                    changed_fraction = (
                        np.count_nonzero(new_support != old_support)
                        / new_support.size
                    )
                    error_dct[name] = np.array(
                        [0.0, 0.0, changed_fraction],
                        dtype=float,
                    )

            self._sigma = np.maximum(
                self._minimum_sigma,
                self._sigma * self.p.sigma_decay,
            )
            self.update_count += 1
            self.curiter += 1

        return error_dct

    def engine_finalize(self):
        """Shrink-wrap has no finalization phase."""
        pass