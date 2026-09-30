"""CPU utilities shared by experimental CDI reconstruction algorithms.

This module provides a small, testable coherent-diffraction-imaging core based
on PtyPy's physical geometry and propagators. It deliberately does not yet
implement a registered PtyPy BaseEngine subclass: ER and HIO are first
validated as numerical reference algorithms before connecting them to the full
PtyScan/ScanModel/Pod lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter

from ptypy.core.geometry import Geo


Array = np.ndarray


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Validate and normalize a two-dimensional NumPy array shape."""
    if len(shape) != 2:
        raise ValueError("shape must contain exactly two dimensions.")

    rows = int(shape[0])
    columns = int(shape[1])

    if rows <= 0 or columns <= 0:
        raise ValueError("shape dimensions must be positive.")

    return rows, columns


def _center_from_shape(
    shape: tuple[int, int],
    center_px: tuple[float, float] | None,
) -> tuple[float, float]:
    """Return a centre in NumPy coordinates: ``(row, column)``."""
    rows, columns = _validate_shape(shape)

    if center_px is None:
        return (rows - 1) / 2.0, (columns - 1) / 2.0

    if len(center_px) != 2:
        raise ValueError("center_px must contain exactly two coordinates.")

    return float(center_px[0]), float(center_px[1])


def circular_support(
    shape: tuple[int, int],
    radius_px: float,
    center_px: tuple[float, float] | None = None,
) -> Array:
    """Return a boolean circular real-space support mask.

    Parameters
    ----------
    shape
        Output shape in NumPy order: ``(rows, columns)``.
    radius_px
        Radius of the allowed object region in pixels.
    center_px
        Circle centre in ``(row, column)`` coordinates. If omitted, use the
        geometric centre of the array.
    """
    rows, columns = _validate_shape(shape)

    if radius_px <= 0:
        raise ValueError("radius_px must be positive.")

    center_row, center_column = _center_from_shape(
        (rows, columns),
        center_px,
    )

    row, column = np.indices((rows, columns), dtype=float)

    return (
        (row - center_row) ** 2
        + (column - center_column) ** 2
        <= float(radius_px) ** 2
    )


def rectangular_support(
    shape: tuple[int, int],
    oversampling: float | tuple[float, float],
    center_px: tuple[float, float] | None = None,
) -> Array:
    """Return a rectangular support inferred from real-space oversampling.

    The oversampling ratio is defined independently along each array axis:

    .. math::

        \\sigma_y = N_y / n_y,
        \\qquad
        \\sigma_x = N_x / n_x,

    where ``(N_y, N_x)`` is the computational-array shape and
    ``(n_y, n_x)`` is the support size. Therefore a support of approximate
    size ``(N_y / sigma_y, N_x / sigma_x)`` is constructed.

    Parameters
    ----------
    shape
        Computational-array shape: ``(rows, columns)``.
    oversampling
        Positive scalar for both axes, or ``(sigma_rows, sigma_columns)``.
        Values must be at least one.
    center_px
        Rectangle centre in ``(row, column)`` coordinates. If omitted, use
        the geometric centre of the array.
    """
    rows, columns = _validate_shape(shape)

    if np.isscalar(oversampling):
        sigma_rows = float(oversampling)
        sigma_columns = float(oversampling)
    else:
        if len(oversampling) != 2:
            raise ValueError(
                "oversampling must be a scalar or contain exactly two values."
            )

        sigma_rows = float(oversampling[0])
        sigma_columns = float(oversampling[1])

    if sigma_rows <= 0 or sigma_columns <= 0:
        raise ValueError("oversampling values must be positive.")

    if sigma_rows < 1 or sigma_columns < 1:
        raise ValueError(
            "oversampling must be at least 1 along each axis."
        )

    support_rows = max(1, int(np.round(rows / sigma_rows)))
    support_columns = max(1, int(np.round(columns / sigma_columns)))

    center_row, center_column = _center_from_shape(
        (rows, columns),
        center_px,
    )

    row_start = int(np.floor(center_row - support_rows / 2.0 + 0.5))
    column_start = int(np.floor(center_column - support_columns / 2.0 + 0.5))

    row_start = max(0, min(row_start, rows - support_rows))
    column_start = max(0, min(column_start, columns - support_columns))

    support = np.zeros((rows, columns), dtype=bool)
    support[
        row_start:row_start + support_rows,
        column_start:column_start + support_columns,
    ] = True

    return support


def support_from_array(
    support: Array,
    *,
    shape: tuple[int, int] | None = None,
    threshold: float = 0.5,
) -> Array:
    """Convert a supplied binary, real, or complex array into a boolean support.

    Boolean arrays are copied as masks. Numeric arrays use
    ``support > threshold``. Complex arrays use ``abs(support) > threshold``.

    Parameters
    ----------
    support
        Two-dimensional external support array.
    shape
        Optional required support shape.
    threshold
        Threshold for real and complex numeric masks.
    """
    support = np.asarray(support)

    if support.ndim != 2:
        raise ValueError("support must be a two-dimensional array.")

    if shape is not None:
        expected_shape = _validate_shape(shape)

        if support.shape != expected_shape:
            raise ValueError(
                "support shape must match the requested shape; "
                f"got {support.shape} and {expected_shape}."
            )

    if support.dtype == bool:
        output = support.copy()
    elif np.iscomplexobj(support):
        output = np.abs(support) > threshold
    else:
        if not np.all(np.isfinite(support)):
            raise ValueError("numeric support must contain only finite values.")

        output = support > threshold

    if not np.any(output):
        raise ValueError("support must contain at least one True pixel.")

    return output.astype(bool, copy=False)


def support_from_npy(
    path: str,
    *,
    shape: tuple[int, int] | None = None,
    threshold: float = 0.5,
) -> Array:
    """Load a support mask from a NumPy ``.npy`` file."""
    if not str(path).lower().endswith(".npy"):
        raise ValueError("support file must use the '.npy' extension.")

    loaded = np.load(path, allow_pickle=False)

    return support_from_array(
        loaded,
        shape=shape,
        threshold=threshold,
    )


def support_from_autocorrelation(
    diffraction_intensity: Array,
    *,
    valid_mask: Array | None = None,
    gaussian_sigma_px: float | tuple[float, float] | None = 1.0,
    threshold: float = 0.1,
    closing_iterations: int = 0,
) -> Array:
    """Estimate a binary initial support from diffraction autocorrelation.

    The processing sequence is:

    1. inverse Fourier transform of far-field intensity;
    2. magnitude of the autocorrelation;
    3. optional Gaussian smoothing;
    4. relative thresholding;
    5. optional binary morphological closing.

    Missing detector pixels are filled with zero for this initial estimate
    only; they are not interpreted as physically measured zero intensity.

    Parameters
    ----------
    diffraction_intensity
        Two-dimensional nonnegative far-field intensity.
    valid_mask
        Boolean detector mask; ``True`` identifies measured pixels. If
        omitted, every detector pixel is used.
    gaussian_sigma_px
        Gaussian standard deviation in pixels. A scalar applies along both
        axes, and a two-value tuple specifies ``(sigma_rows, sigma_columns)``.
        ``None`` disables smoothing.
    threshold
        Relative threshold in the open interval ``(0, 1)``.
    closing_iterations
        Number of binary-closing iterations. Zero disables closing.

    Returns
    -------
    numpy.ndarray
        Boolean support-like mask with the same shape as the input intensity.
    """
    intensity = np.asarray(diffraction_intensity)

    if intensity.ndim != 2:
        raise ValueError(
            "diffraction_intensity must be a two-dimensional array."
        )

    if not np.all(np.isfinite(intensity)):
        raise ValueError(
            "diffraction_intensity must contain only finite values."
        )

    if np.any(intensity < 0):
        raise ValueError(
            "diffraction_intensity must be non-negative."
        )

    if valid_mask is None:
        valid_mask = np.ones(intensity.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

        if valid_mask.shape != intensity.shape:
            raise ValueError(
                "valid_mask and diffraction_intensity must have identical "
                f"shapes; got {valid_mask.shape} and {intensity.shape}."
            )

    if not np.any(valid_mask):
        raise ValueError(
            "valid_mask must contain at least one True pixel."
        )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must lie strictly between 0 and 1."
        )

    if gaussian_sigma_px is not None:
        sigma = np.asarray(gaussian_sigma_px, dtype=float)

        if sigma.ndim > 1 or sigma.size not in {1, 2}:
            raise ValueError(
                "gaussian_sigma_px must be a scalar, a two-value tuple, "
                "or None."
            )

        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError(
                "gaussian_sigma_px must contain positive finite values."
            )

    if closing_iterations < 0:
        raise ValueError(
            "closing_iterations must be non-negative."
        )

    filled_intensity = np.zeros_like(intensity, dtype=np.float64)
    filled_intensity[valid_mask] = intensity[valid_mask]

    autocorrelation = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(filled_intensity),
            norm="ortho",
        )
    )

    autocorrelation_magnitude = np.abs(autocorrelation)

    if gaussian_sigma_px is not None:
        autocorrelation_magnitude = gaussian_filter(
            autocorrelation_magnitude,
            sigma=gaussian_sigma_px,
        )

    maximum = float(autocorrelation_magnitude.max())

    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(
            "autocorrelation has no positive finite maximum."
        )

    support = (
        autocorrelation_magnitude
        > threshold * maximum
    )

    if closing_iterations > 0:
        support = binary_closing(
            support,
            iterations=closing_iterations,
        )

    if not np.any(support):
        raise ValueError(
            "thresholding produced an empty autocorrelation support."
        )

    return np.asarray(support, dtype=bool)


def masked_modulus_projection(
    detector_field: Array,
    measured_amplitude: Array,
    valid_mask: Array,
    *,
    epsilon: float = 1e-12,
) -> Array:
    """Replace detector amplitudes at valid pixels while retaining phase.

    At valid detector pixels, the result is:

    .. math::

        A_{\\mathrm{meas}} \\exp[i\\arg(\\Psi)].

    At invalid pixels the field is not changed. Invalid pixels represent
    unmeasured regions such as a beamstop, detector gap, dead pixel, or a
    saturated pixel not modelled by a censored-data likelihood.

    ``valid_mask=True`` means that the measured Fourier-modulus constraint is
    enforced at that pixel.
    """
    detector_field = np.asarray(detector_field)
    measured_amplitude = np.asarray(measured_amplitude)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if detector_field.shape != measured_amplitude.shape:
        raise ValueError(
            "detector_field and measured_amplitude must have identical "
            f"shapes; got {detector_field.shape} and "
            f"{measured_amplitude.shape}."
        )

    if detector_field.shape != valid_mask.shape:
        raise ValueError(
            "detector_field and valid_mask must have identical shapes; "
            f"got {detector_field.shape} and {valid_mask.shape}."
        )

    if not np.iscomplexobj(detector_field):
        raise TypeError("detector_field must be complex-valued.")

    if not np.all(np.isfinite(measured_amplitude)):
        raise ValueError("measured_amplitude must contain only finite values.")

    if np.any(measured_amplitude < 0):
        raise ValueError("measured_amplitude must be non-negative.")

    magnitude = np.abs(detector_field)

    phase = np.ones_like(detector_field, dtype=np.complex128)
    nonzero = magnitude > epsilon
    phase[nonzero] = detector_field[nonzero] / magnitude[nonzero]

    projected = detector_field.astype(np.complex128, copy=True)
    projected[valid_mask] = (
        measured_amplitude[valid_mask] * phase[valid_mask]
    )

    return projected


def masked_amplitude_error(
    detector_field: Array,
    measured_amplitude: Array,
    valid_mask: Array,
) -> float:
    """Return normalized L2 Fourier-amplitude residual on valid pixels."""
    detector_field = np.asarray(detector_field)
    measured_amplitude = np.asarray(measured_amplitude)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if detector_field.shape != measured_amplitude.shape:
        raise ValueError("detector_field and measured_amplitude shape mismatch.")

    if detector_field.shape != valid_mask.shape:
        raise ValueError("detector_field and valid_mask shape mismatch.")

    if not np.any(valid_mask):
        raise ValueError("valid_mask must contain at least one True pixel.")

    residual = (
        np.abs(detector_field[valid_mask])
        - measured_amplitude[valid_mask]
    )

    denominator = np.linalg.norm(measured_amplitude[valid_mask])

    if denominator == 0.0:
        return float(np.linalg.norm(residual))

    return float(np.linalg.norm(residual) / denominator)


@dataclass
class CDIGeometry:
    """Thin CPU CDI wrapper around PtyPy ``Geo`` and its propagator."""

    geo: Geo

    @classmethod
    def from_parameters(
        cls,
        *,
        shape: tuple[int, int],
        energy_kev: float,
        distance_m: float,
        detector_psize_m: float | tuple[float, float],
        propagation: str = "farfield",
        ffttype: str = "scipy",
        center: str | tuple[float, float] = "fftshift",
        origin: str | tuple[float, float] = "fftshift",
        data_type: type = np.complex128,
    ) -> "CDIGeometry":
        """Create a physical PtyPy ``Geo`` object from CDI parameters."""
        _validate_shape(shape)

        if propagation not in {"farfield", "nearfield"}:
            raise ValueError(
                "propagation must be either 'farfield' or 'nearfield'."
            )

        pars = {
            "shape": shape,
            "energy": energy_kev,
            "distance": distance_m,
            "psize": detector_psize_m,
            "propagation": propagation,
            "ffttype": ffttype,
            "center": center,
            "origin": origin,
        }

        geo = Geo(pars=pars)

        # A stand-alone Geo has no Ptycho owner. PtyPy therefore creates its
        # initial propagator in complex64. Recreate the propagator explicitly
        # in the requested precision for reproducible CDI reference tests.
        propagator_class = type(geo.propagator)
        geo._propagator = propagator_class(
            geo.p,
            ffttype=ffttype,
            dtype=data_type,
        )

        return cls(geo=geo)

    @property
    def shape(self) -> tuple[int, int]:
        """Object- and detector-field shape."""
        return tuple(int(value) for value in self.geo.shape)

    @property
    def object_psize_m(self) -> tuple[float, float]:
        """Object-plane pixel size in metres."""
        return tuple(float(value) for value in self.geo.resolution)

    @property
    def detector_psize_m(self) -> tuple[float, float]:
        """Detector-plane pixel size in metres."""
        return tuple(float(value) for value in self.geo.psize)

    def forward(self, object_field: Array) -> Array:
        """Propagate a complex object-plane field to the detector plane."""
        object_field = np.asarray(object_field)

        if object_field.shape != self.shape:
            raise ValueError(
                "object_field shape must match geometry shape; "
                f"got {object_field.shape} and {self.shape}."
            )

        return self.geo.propagator.fw(object_field)

    def backward(self, detector_field: Array) -> Array:
        """Back-propagate a complex detector-plane field to object space."""
        detector_field = np.asarray(detector_field)

        if detector_field.shape != self.shape:
            raise ValueError(
                "detector_field shape must match geometry shape; "
                f"got {detector_field.shape} and {self.shape}."
            )

        return self.geo.propagator.bw(detector_field)


@dataclass
class CDIProblem:
    """Measured CDI data, detector mask, support, and physical geometry."""

    geometry: CDIGeometry
    measured_intensity: Array
    valid_mask: Array
    support: Array

    def __post_init__(self) -> None:
        intensity = np.asarray(self.measured_intensity)
        valid_mask = np.asarray(self.valid_mask, dtype=bool)
        support = np.asarray(self.support, dtype=bool)

        shape = self.geometry.shape

        if intensity.shape != shape:
            raise ValueError(
                "measured_intensity shape must match geometry shape; "
                f"got {intensity.shape} and {shape}."
            )

        if valid_mask.shape != shape:
            raise ValueError(
                "valid_mask shape must match geometry shape; "
                f"got {valid_mask.shape} and {shape}."
            )

        if support.shape != shape:
            raise ValueError(
                "support shape must match geometry shape; "
                f"got {support.shape} and {shape}."
            )

        if not np.all(np.isfinite(intensity)):
            raise ValueError("measured_intensity must contain only finite values.")

        if np.any(intensity < 0):
            raise ValueError("measured_intensity must be non-negative.")

        if not np.any(valid_mask):
            raise ValueError("valid_mask must contain at least one True pixel.")

        if not np.any(support):
            raise ValueError("support must contain at least one True pixel.")

        self.measured_intensity = intensity.astype(np.float64, copy=False)
        self.valid_mask = valid_mask
        self.support = support
        self.measured_amplitude = np.sqrt(self.measured_intensity)

    def data_projection(
        self,
        object_field: Array,
    ) -> tuple[Array, float]:
        """Apply detector modulus projection and return real-space candidate."""
        detector_field = self.geometry.forward(object_field)

        amplitude_error = masked_amplitude_error(
            detector_field=detector_field,
            measured_amplitude=self.measured_amplitude,
            valid_mask=self.valid_mask,
        )

        constrained_detector_field = masked_modulus_projection(
            detector_field=detector_field,
            measured_amplitude=self.measured_amplitude,
            valid_mask=self.valid_mask,
        )

        candidate = self.geometry.backward(constrained_detector_field)

        return candidate, amplitude_error