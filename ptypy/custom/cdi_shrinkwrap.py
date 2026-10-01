"""Shrink-wrap support updates for custom CDI reconstructions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter
from ptypy.custom.cdi_common import register_citation

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


@dataclass
class ShrinkWrapSupport:
    """Update a CDI real-space support from the current object estimate.

    A support update follows the shrink-wrap procedure:

    1. calculate the modulus of the current complex object field;
    2. blur that modulus with a Gaussian kernel;
    3. threshold relative to the maximum blurred modulus;
    4. optionally apply binary morphological closing;
    5. decay the Gaussian width for the next call.

    The default values reproduce the schedule reported by Marchesini et al.
    (2003): sigma starts at 3 px, is reduced by 1 percent after each support
    update, and is bounded below by 1.5 px. The threshold is 20 percent of
    the maximum blurred amplitude.

    Parameters
    ----------
    gaussian_sigma_px
        Initial Gaussian standard deviation in pixels. A scalar applies to
        both axes; a two-element tuple sets ``(sigma_rows, sigma_columns)``.
    threshold
        Relative threshold in the open interval ``(0, 1)``.
    sigma_decay
        Multiplicative factor applied after every update. ``0.99`` means a
        1 percent reduction per support update.
    minimum_sigma_px
        Minimum Gaussian width in pixels. A scalar applies to both axes; a
        two-element tuple sets ``(sigma_rows, sigma_columns)``.
    closing_iterations
        Number of binary-closing iterations after thresholding. Zero disables
        the operation. This is an optional robustness extension, not part of
        the minimal published shrink-wrap update.
    """
    gaussian_sigma_px: float | tuple[float, float] = 3.0
    threshold: float = 0.20
    sigma_decay: float = 0.99
    minimum_sigma_px: float | tuple[float, float] = 1.5
    closing_iterations: int = 0

    def __post_init__(self) -> None:
        """Validate parameters and initialize mutable sigma state."""
        self._sigma = self._validate_sigma(
            self.gaussian_sigma_px,
            parameter_name="gaussian_sigma_px",
        )

        self._minimum_sigma = self._validate_sigma(
            self.minimum_sigma_px,
            parameter_name="minimum_sigma_px",
        )

        if np.any(self._minimum_sigma > self._sigma):
            raise ValueError(
                "minimum_sigma_px must not exceed gaussian_sigma_px."
            )

        if not 0.0 < self.threshold < 1.0:
            raise ValueError(
                "threshold must lie strictly between 0 and 1."
            )

        if not 0.0 < self.sigma_decay <= 1.0:
            raise ValueError(
                "sigma_decay must lie in the interval (0, 1]."
            )

        if isinstance(self.closing_iterations, bool):
            raise ValueError(
                "closing_iterations must be a non-negative integer."
            )

        if (
            not isinstance(self.closing_iterations, int)
            or self.closing_iterations < 0
        ):
            raise ValueError(
                "closing_iterations must be a non-negative integer."
            )

        register_citation(SHRINKWRAP_ARTICLE)

        self.update_count = 0

    @staticmethod
    def _validate_sigma(
        sigma: float | tuple[float, float],
        *,
        parameter_name: str,
    ) -> Array:
        """Return validated sigma as a length-two float array."""
        sigma_array = np.asarray(sigma, dtype=float)

        if sigma_array.ndim > 1 or sigma_array.size not in {1, 2}:
            raise ValueError(
                f"{parameter_name} must be a scalar or a two-value tuple."
            )

        if sigma_array.size == 1:
            sigma_array = np.repeat(sigma_array, 2)

        if (
            not np.all(np.isfinite(sigma_array))
            or np.any(sigma_array <= 0.0)
        ):
            raise ValueError(
                f"{parameter_name} must contain positive finite values."
            )

        return sigma_array.astype(np.float64, copy=True)

    @property
    def current_sigma_px(self) -> tuple[float, float]:
        """Gaussian width that will be used by the next update."""
        return (float(self._sigma[0]), float(self._sigma[1]))

    @property
    def minimum_sigma_px_tuple(self) -> tuple[float, float]:
        """Configured lower limit for the Gaussian width."""
        return (
            float(self._minimum_sigma[0]),
            float(self._minimum_sigma[1]),
        )

    def reset(self) -> None:
        """Restore the initial sigma and clear the update counter."""
        self._sigma = self._validate_sigma(
            self.gaussian_sigma_px,
            parameter_name="gaussian_sigma_px",
        )
        self.update_count = 0

    def update(
        self,
        object_field: Array,
    ) -> Array:
        """Return an updated boolean support from a complex object estimate.

        The returned support is not automatically written anywhere. This keeps
        the operation composable: the caller decides which ``CDIProblem`` or
        reconstruction state should receive the new mask.

        Parameters
        ----------
        object_field
            Two-dimensional current estimate of the complex object field.

        Returns
        -------
        numpy.ndarray
            Two-dimensional boolean support mask.
        """
        object_field = np.asarray(object_field)

        if object_field.ndim != 2:
            raise ValueError(
                "object_field must be a two-dimensional array."
            )

        if not np.all(np.isfinite(object_field)):
            raise ValueError(
                "object_field must contain finite values."
            )

        amplitude = np.abs(object_field)

        blurred_amplitude = gaussian_filter(
            amplitude,
            sigma=tuple(self._sigma),
        )

        maximum = float(blurred_amplitude.max())

        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError(
                "blurred object amplitude has no positive finite maximum."
            )

        support = blurred_amplitude >= self.threshold * maximum

        if self.closing_iterations > 0:
            support = binary_closing(
                support,
                iterations=self.closing_iterations,
            )

        support = np.asarray(support, dtype=bool)

        if not np.any(support):
            raise RuntimeError(
                "Shrink-wrap thresholding produced an empty support."
            )

        self._sigma = np.maximum(
            self._minimum_sigma,
            self._sigma * self.sigma_decay,
        )

        self.update_count += 1

        return support