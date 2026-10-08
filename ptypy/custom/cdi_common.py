"""CPU utilities shared by the custom CDI reconstruction algorithms.

Provides support constructors, detector-domain projections, a thin wrapper
around PtyPy ``Geo`` and its propagator, and ``CDIScan``, which feeds a single
far-field diffraction pattern into a PtyPy ``Ptycho`` instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter, map_coordinates, binary_erosion
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt
from ptypy.utils.verbose import logger
from ptypy.utils import parallel

from ptypy.core.data import PtyScan
from ptypy.core.geometry import Geo
from ptypy.core.manager import Full, Vanilla
from ptypy.engines.base import BaseEngine
from ptypy.experiment import register as register_ptyscan

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

    def set_support(self, support: Array) -> None:
        """Replace the real-space support after validating it."""
        support = np.asarray(support, dtype=bool)

        if support.shape != self.geometry.shape:
            raise ValueError(
                "support shape must match geometry shape; "
                f"got {support.shape} and {self.geometry.shape}."
            )

        if not np.any(support):
            raise ValueError("support must contain at least one True pixel.")

        self.support = support.copy()

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


def estimate_cdi_center(intensity, valid_mask=None, *, method="mass",
                        initial_center=None, search_radius=8.0,
                        tolerance=1e-4, min_overlap=0.5):
    """Estimate a raw-frame center and return numerical diagnostics.

    CDIScan rounds this estimate before passing it to the geometry.
    Optimizer tolerance is not a physical accuracy guarantee.
    """
    image = np.asarray(intensity, dtype=np.float64)
    mask = np.ones(image.shape, bool) if valid_mask is None else np.asarray(valid_mask, bool)
    if image.ndim != 2 or mask.shape != image.shape:
        raise ValueError("Intensity and mask must be matching 2D arrays.")
    if not mask.any() or not np.isfinite(image[mask]).all() or (image[mask] < 0).any():
        raise ValueError("Valid intensities must be finite and nonnegative; mask cannot be empty.")
    scale = image[mask].max()
    if scale <= 0:
        raise ValueError("No positive measured intensity.")
    image = np.where(mask, image / scale, 0.0)
    if method == "mass":
        y, x = np.indices(image.shape, dtype=float)
        center = np.array([(y*image).sum(), (x*image).sum()]) / image.sum()
        return tuple(center), dict(method=method, scores=(), boundary_hit=False)
    if method not in ("inversion", "reflections"):
        raise ValueError("Unknown autocenter method.")
    initial = (np.array(image.shape)-1)/2 if initial_center is None else np.asarray(initial_center, float)
    radius = np.broadcast_to(np.asarray(search_radius, float), (2,))
    if initial.shape != (2,) or not np.isfinite(initial).all() or not np.isfinite(radius).all() or (radius <= 0).any():
        raise ValueError("Invalid initial center or radius.")
    if (initial < 0).any() or (initial > np.array(image.shape)-1).any():
        raise ValueError("Initial center is outside the frame.")
    if not 0 < min_overlap <= 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid overlap or tolerance.")
    y, x = np.nonzero(mask)
    positions = np.vstack([y, x]).astype(float)
    measured = image[y, x]
    groups = [(0, 1)] if method == "inversion" else [(0,), (1,)]
    answer = initial.copy()
    diagnostics = []
    eroded = binary_erosion(mask, structure=np.ones((3,3)), iterations=2)
    for axes in groups:
        lower = np.maximum(0, initial-radius)
        upper = np.minimum(np.array(image.shape)-1, initial+radius)
        grids = [np.arange(np.ceil(2*lower[a]), np.floor(2*upper[a])+1)/2 for a in axes]
        best = None
        for values in product(*grids):
            candidate = initial.copy()
            candidate[list(axes)] = values
            mapped = positions.copy()
            for a in axes:
                mapped[a] = 2*candidate[a]-positions[a]
            iy, ix = np.rint(mapped).astype(int)
            inside = (iy>=0)&(iy<image.shape[0])&(ix>=0)&(ix<image.shape[1])
            valid = inside.copy()
            valid[inside] &= mask[iy[inside],ix[inside]]
            if valid.sum() < max(8, min_overlap*mask.sum()):
                continue
            score = _cdi_pair_correlation(measured[valid], image[iy[valid],ix[valid]])
            if not np.isfinite(score):
                continue
            key = (score, valid.sum(), -np.sum((candidate-initial)**2))
            if best is None or key > best[0]:
                best = (key, candidate)
        if best is None:
            raise ValueError("No symmetry candidate with sufficient overlap and contrast.")
        coarse = best[1]
        mapped = positions.copy()
        for a in axes:
            mapped[a] = 2*coarse[a]-positions[a]
        safe = map_coordinates(eroded.astype(float), mapped, order=0, mode="constant", cval=0)>0.5
        if safe.sum() < max(8, min_overlap*mask.sum()):
            raise ValueError("Insufficient valid pairs for subpixel refinement.")
        pos = positions[:,safe]
        target = measured[safe]
        bounds = [(max(lower[a],coarse[a]-0.5), min(upper[a],coarse[a]+0.5)) for a in axes]
        def objective(values):
            mapped = pos.copy()
            for a, value in zip(axes, values):
                mapped[a] = 2*value-pos[a]
            sampled = map_coordinates(image, mapped, order=1, mode="constant", cval=0)
            correlation = _cdi_pair_correlation(target, sampled)
            return 2.0 if not np.isfinite(correlation) else 1.0-correlation
        result = minimize(objective, coarse[list(axes)], method="Powell", bounds=bounds,
                          options=dict(xtol=tolerance, ftol=1e-10, maxiter=150))
        if not result.success:
            raise ValueError("Center refinement did not converge: " + str(result.message))
        answer[list(axes)] = result.x
        hit = any(abs(v-lower[a]) < 0.01 or abs(v-upper[a]) < 0.01 for a,v in zip(axes,result.x))
        diagnostics.append(dict(axes=axes, score=1-result.fun, overlap=float(safe.mean()),
                                coarse_center=tuple(coarse), boundary_hit=hit))
    return tuple(answer), dict(method=method, searches=diagnostics,
                              boundary_hit=any(item["boundary_hit"] for item in diagnostics))


def _cdi_pair_correlation(a, b):
    """Return normalized correlation for valid paired samples."""
    a = a-a.mean()
    b = b-b.mean()
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/norm) if norm > 1e-14 else float("nan")

def sigma_clip_background(
    values,
    sigma=3.0,
    max_iter=5,
    eps=1e-12,
):
    """Estimate background using median/MAD clipping and retained mean."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]

    sigma = float(sigma)
    eps = float(eps)

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite.")

    if isinstance(max_iter, (bool, np.bool_)):
        raise ValueError("max_iter must be a nonnegative integer.")

    if int(max_iter) != max_iter or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer.")

    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite.")

    if arr.size == 0:
        raise ValueError("No finite pixels available for background estimation.")

    mask = np.ones(arr.shape, dtype=bool)

    for _ in range(int(max_iter)):
        subset = arr[mask]

        if subset.size == 0:
            break

        median = np.median(subset)
        mad = np.median(np.abs(subset - median))
        sigma_bg = 1.4826 * mad

        if sigma_bg <= eps:
            break

        new_mask = np.abs(arr - median) < sigma * sigma_bg

        if np.array_equal(new_mask, mask):
            break

        mask = new_mask

    if not np.any(mask):
        return float(np.median(arr))

    return float(arr[mask].mean())


def butterworth_filter(
    intensity,
    valid_mask,
    *,
    mode="lowpass",
    cutoff_frequency_ratio=0.08,
    order=2,
    npad=32,
    mask_policy="error",
):
    """Filter detector intensity using a squared Butterworth response."""
    cutoff = float(cutoff_frequency_ratio)
    order = float(order)

    if mode not in {"lowpass", "highpass"}:
        raise ValueError("mode must be 'lowpass' or 'highpass'.")

    if not np.isfinite(cutoff) or not 0 < cutoff <= 0.5:
        raise ValueError(
            "cutoff_frequency_ratio must be finite and in (0, 0.5]."
        )

    if not np.isfinite(order) or order <= 0:
        raise ValueError("order must be positive and finite.")

    if isinstance(npad, (bool, np.bool_)):
        raise ValueError("npad must be a nonnegative integer.")

    if int(npad) != npad or npad < 0:
        raise ValueError("npad must be a nonnegative integer.")

    if mask_policy not in {"error", "nearest"}:
        raise ValueError("mask_policy must be 'error' or 'nearest'.")

    npad = int(npad)
    working = intensity.copy()

    if not np.all(valid_mask):
        if mask_policy == "error":
            raise ValueError(
                "Butterworth filtering requires an explicit missing-pixel "
                "policy. Use mask_policy='nearest' to fill missing pixels "
                "temporarily for filtering."
            )

        nearest = distance_transform_edt(
            ~valid_mask,
            return_distances=False,
            return_indices=True,
        )
        working[~valid_mask] = working[tuple(nearest[:, ~valid_mask])]

    if npad:
        working = np.pad(working, npad, mode="edge")

    rows, columns = working.shape
    fy = np.fft.fftfreq(rows)[:, None]
    fx = np.fft.rfftfreq(columns)[None, :]
    radius = np.hypot(fy, fx)

    with np.errstate(over="ignore"):
        response = 1.0 / (1.0 + (radius / cutoff) ** (2.0 * order))

    spectrum = np.fft.rfft2(working)
    lowpass = np.fft.irfft2(
        spectrum * response,
        s=working.shape,
    )

    if npad:
        lowpass = lowpass[npad:-npad, npad:-npad]

    if mode == "lowpass":
        return lowpass

    return intensity - lowpass


def denoise_diffraction_frame(intensity, valid_mask, pars):
    """Return processed intensity without changing inputs or detector mask."""
    source = np.asarray(intensity)

    if source.ndim != 2 or np.iscomplexobj(source):
        raise ValueError("intensity must be a real two-dimensional array.")

    output = source.astype(np.float64, copy=True)

    if not bool(pars.get("active", False)):
        return output

    if valid_mask is None:
        valid = np.ones(output.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)

    if valid.shape != output.shape:
        raise ValueError("valid_mask must have the same shape as intensity.")

    if not np.any(valid):
        raise ValueError("valid_mask contains no valid pixels.")

    if not np.all(np.isfinite(output[valid])):
        raise ValueError("Valid intensity pixels must be finite.")

    if np.any(output[valid] < 0):
        raise ValueError("Valid input intensity pixels must be nonnegative.")

    name = pars.get("name", "butterworth")
    background = None

    if name == "sigma_clip":
        selected = valid.copy()
        statistics_mask = pars.get("statistics_mask", None)

        if statistics_mask is not None:
            statistics_mask = np.asarray(statistics_mask)

            if statistics_mask.shape != output.shape:
                raise ValueError(
                    "statistics_mask must have the same shape as intensity."
                )

            if statistics_mask.dtype != np.dtype(bool):
                raise TypeError("statistics_mask must be a boolean array.")

            selected &= statistics_mask

        if not np.any(selected):
            raise ValueError(
                "statistics_mask selects no valid pixels."
            )

        background = sigma_clip_background(
            output[selected],
            sigma=pars.get("sigma", 3.0),
            max_iter=pars.get("max_iter", 5),
        )
        processed = output - background

    elif name == "butterworth":
        processed = butterworth_filter(
            output,
            valid,
            mode=pars.get("mode", "lowpass"),
            cutoff_frequency_ratio=pars.get(
                "cutoff_frequency_ratio", 0.08
            ),
            order=pars.get("order", 2),
            npad=pars.get("npad", 32),
            mask_policy=pars.get("mask_policy", "error"),
        )

    else:
        raise ValueError(
            "denoising.name must be 'sigma_clip' or 'butterworth'."
        )

    if not np.all(np.isfinite(processed[valid])):
        raise ValueError("Denoising produced nonfinite valid intensities.")

    clipped_fraction = float(np.mean(processed[valid] < 0))
    output[valid] = np.maximum(processed[valid], 0.0)

    # Invalid pixels remain unmeasured, not reconstructed by denoising.
    output[~valid] = 0.0

    if bool(pars.get("verbose", False)):
        message = (
            f"CDI denoising: method={name}, "
            f"valid_sum_before={source[valid].sum(dtype=np.float64):.6g}, "
            f"valid_sum_after={output[valid].sum():.6g}, "
            f"clipped_fraction={clipped_fraction:.6g}"
        )

        if background is not None:
            message += f", background={background:.6g}"

        logger.info(message)

    return output


@register_ptyscan("CDIScan")
class CDIScan(PtyScan):
    """
    Single far-field CDI diffraction pattern supplied as an array.

    Defaults:

    [name]
    default = CDIScan
    type = str
    help =

    [intensity]
    default = None
    type = ndarray
    help = Two-dimensional non-negative far-field intensity

    [mask]
    default = None
    type = ndarray
    help = Boolean detector mask, True for valid pixels; None means all valid

    [autocenter]
    default = None
    type = bool, None
    help = Estimate the diffraction center automatically when center is None

    [autocenter_method]
    default = mass
    type = str
    help = CDI center estimator
    choices = mass, inversion, reflections

    [autocenter_initial_center]
    default = None
    type = tuple, list, ndarray, None
    help = Initial center in raw row, column coordinates

    [autocenter_search_radius]
    default = 8.0
    type = float
    help = Center search radius in raw pixels

    [autocenter_tolerance]
    default = 0.0001
    type = float
    help = Optimizer tolerance, not an accuracy guarantee

    [autocenter_min_overlap]
    default = 0.5
    type = float
    help = Minimum valid pair fraction

    [support]
    default = None
    type = Param
    help = Initial real-space support, computed once when the data are loaded

    [support.kind]
    default = autocorrelation
    type = str
    help = One of autocorrelation, circle, file, array

    [support.threshold]
    default = 0.04
    type = float
    help = Relative autocorrelation threshold (kind = autocorrelation)

    [support.blur_sigma]
    default = None
    type = float
    help = Gaussian blur of the autocorrelation in pixels; None disables it (kind = autocorrelation)

    [support.closing_iterations]
    default = 0
    type = int
    help = Binary closing iterations after thresholding (kind = autocorrelation)

    [support.radius]
    default = None
    type = float
    help = Circle radius in pixels (kind = circle)

    [support.file]
    default = None
    type = str
    help = Path to a .npy support mask (kind = file)

    [support.array]
    default = None
    type = ndarray
    help = Support mask supplied as an array (kind = array)

    [denoising]
    default =
    type = Param
    help = Optional detector-intensity denoising before spatial preprocessing

    [denoising.active]
    default = False
    type = bool
    help = Enable detector-intensity denoising

    [denoising.name]
    default = butterworth
    type = str
    help = Denoising method: butterworth or sigma_clip

    [denoising.mode]
    default = lowpass
    type = str
    help = Butterworth mode: lowpass or highpass

    [denoising.cutoff_frequency_ratio]
    default = 0.08
    type = float
    help = Butterworth cutoff in cycles per detector pixel

    [denoising.order]
    default = 2.0
    type = float
    help = Positive Butterworth order

    [denoising.npad]
    default = 32
    type = int
    help = Edge padding width before Butterworth filtering

    [denoising.mask_policy]
    default = error
    type = str
    help = Missing-pixel handling for Butterworth: error or nearest

    [denoising.sigma]
    default = 3.0
    type = float
    help = Sigma-clipping multiplier applied to the MAD-based scale

    [denoising.max_iter]
    default = 5
    type = int
    help = Maximum sigma-clipping iterations

    [denoising.statistics_mask]
    default = None
    type = ndarray
    help = Boolean raw-frame mask, True selects pixels for background estimation

    [denoising.verbose]
    default = False
    type = bool
    help = Log processing statistics without storing diagnostics

    """

    def __init__(self, pars=None, **kwargs):
        p = self.DEFAULT.copy(depth=99)
        p.update(pars, in_place_depth=99)
        p.update(kwargs, in_place_depth=99)

        if p.intensity is None:
            raise ValueError("CDIScan requires data.intensity.")

        intensity = np.asarray(p.intensity, dtype=np.float64)

        if intensity.ndim != 2:
            raise ValueError("data.intensity must be two-dimensional.")

        if not np.all(np.isfinite(intensity)) or np.any(intensity < 0):
            raise ValueError("data.intensity must be finite and non-negative.")

        if p.mask is None:
            mask = np.ones(intensity.shape, dtype=bool)
        else:
            mask = np.asarray(p.mask, dtype=bool)

            if mask.shape != intensity.shape:
                raise ValueError(
                    "data.mask must have the same shape as data.intensity."
                )

        p.shape = intensity.shape
        p.numframes = 1

        super().__init__(p)

        self._intensity = intensity
        self._mask = mask
        self.support = self._initial_support(p.support)

    def _mpi_autocenter(self, data, weights):
        """Return an integer center using the existing preprocessing hook."""
        if self.info.autocenter_method == "mass":
            center = super()._mpi_autocenter(data, weights)
            return np.rint(center).astype(int)
        local = {}
        for index, frame in data.items():
            try:
                center, diagnostics = estimate_cdi_center(
                    frame, weights[index] > 0,
                    method=self.info.autocenter_method,
                    initial_center=self.info.autocenter_initial_center,
                    search_radius=self.info.autocenter_search_radius,
                    tolerance=self.info.autocenter_tolerance,
                    min_overlap=self.info.autocenter_min_overlap,
                )
                local[index] = dict(center=center, diagnostics=diagnostics, error=None)
            except Exception as error:
                local[index] = dict(error=str(error))
        gathered = parallel.gather_dict(local)
        payload = None
        if parallel.master:
            errors = [item["error"] for item in gathered.values() if item["error"]]
            if errors or not gathered:
                payload = dict(error="; ".join(errors) or "No diffraction frames.")
            else:
                payload = dict(error=None,
                    center=np.mean([item["center"] for item in gathered.values()], axis=0),
                    diagnostics={key: item["diagnostics"] for key,item in gathered.items()})
        payload = parallel.bcast(payload)
        if payload["error"]:
            raise ValueError("CDI autocenter failed: " + payload["error"])
        self.centering_diagnostics = payload["diagnostics"]
        return np.rint(payload["center"]).astype(int)

    def _initial_support(self, pars):
        """Compute the initial support from the data.support parameters."""
        shape = self._intensity.shape

        if pars.kind == "autocorrelation":
            return support_from_autocorrelation(
                self._intensity,
                valid_mask=self._mask,
                gaussian_sigma_px=pars.blur_sigma,
                threshold=pars.threshold,
                closing_iterations=pars.closing_iterations,
            )

        if pars.kind == "circle":
            if pars.radius is None:
                raise ValueError("data.support.radius is required for kind='circle'.")
            return circular_support(shape=shape, radius_px=pars.radius)

        if pars.kind == "file":
            if pars.file is None:
                raise ValueError("data.support.file is required for kind='file'.")
            return support_from_npy(pars.file, shape=shape)

        if pars.kind == "array":
            if pars.array is None:
                raise ValueError("data.support.array is required for kind='array'.")
            return support_from_array(pars.array, shape=shape)

        raise ValueError(
            "data.support.kind must be 'autocorrelation', 'circle', 'file', or 'array'."
        )

    def load_positions(self):
        return np.zeros((1, 2))

    def load_weight(self):
        return self._mask.copy()

    def load(self, indices):
        return {index: self._intensity.copy() for index in indices}, {}, {}

    def correct(self, raw, weights, common):
        """Apply optional denoising before spatial preprocessing."""
        data, weights = super().correct(raw, weights, common)

        pars = self.info.get("denoising", None)

        if pars is None or not pars.get("active", False):
            return data, weights

        processed = {}

        for index, frame in data.items():
            frame_weight = weights.get(index)

            if frame_weight is None:
                frame_weight = self.weight2d

            if frame_weight is None:
                frame_weight = self._mask

            valid_mask = np.asarray(frame_weight) > 0

            processed[index] = denoise_diffraction_frame(
                frame,
                valid_mask,
                pars,
            )

        return processed, weights

def cdi_support(ptycho, pod) -> Array:
    """Return the current support of the object seen by ``pod``.

    On first access the support is taken from the initial support computed
    by the scan's ``CDIScan``. Later engines read or replace the same entry.
    """
    supports = ptycho.__dict__.setdefault("cdi_supports", {})
    key = pod.ob_view.storageID

    if key not in supports:
        supports[key] = pod.model.ptyscan.support.copy()

    return supports[key]


def set_cdi_support(ptycho, pod, support: Array) -> None:
    """Replace the current support of the object seen by ``pod``."""
    current = cdi_support(ptycho, pod)
    support = np.asarray(support, dtype=bool)

    if support.shape != current.shape:
        raise ValueError(
            f"support shape must be {current.shape}; got {support.shape}."
        )

    if not np.any(support):
        raise ValueError("support must contain at least one True pixel.")

    ptycho.cdi_supports[pod.ob_view.storageID] = support.copy()

def remove_global_phase(
    object_field: Array,
    support: Array,
) -> Array:
    """Set the phase of the complex object sum inside support to zero."""
    object_field = np.asarray(object_field)
    support = np.asarray(support, dtype=bool)

    if object_field.ndim != 2:
        raise ValueError("object_field must be a two-dimensional array.")

    if object_field.shape != support.shape:
        raise ValueError(
            "object_field and support must have identical shapes."
        )

    if not np.any(support):
        raise ValueError("support must contain at least one True pixel.")

    reference = object_field[support].sum()

    if not np.isfinite(reference) or np.abs(reference) == 0.0:
        raise ValueError(
            "object sum inside support is zero; cannot fix global phase."
        )

    return object_field * np.exp(-1j * np.angle(reference))

class CDIProjectionEngine(BaseEngine):
    """Base class for single-frame CDI engines.

    Each iteration applies the detector-modulus projection through the PtyPy
    propagator of the pod and then a real-space update defined by
    ``object_update``. The probe is set to one because CDI has no probe.

    Defaults:

    [phase_gauge]
    default = True
    type = bool
    help = Remove the global object phase after this engine block
    """

    SUPPORTED_MODELS = [Vanilla, Full]

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        self._amplitudes = {}

    def engine_initialize(self):
        for storage in self.pr.storages.values():
            storage.fill(1.0)

    def engine_prepare(self):
        self._amplitudes = {
            pod_id: np.sqrt(pod.diff) for pod_id, pod in self.pods.items()
        }

    def engine_iterate(self, num=1):
        error_dct = {}

        for _ in range(num):
            for name, diff_view in self.di.views.items():
                if not diff_view.active:
                    continue

                for pod_id, pod in diff_view.pods.items():
                    current = pod.object.copy()
                    amplitude = self._amplitudes[pod_id]
                    detector_field = pod.fw(pod.probe * current)

                    amplitude_error = masked_amplitude_error(
                        detector_field, amplitude, pod.mask
                    )
                    projected = masked_modulus_projection(
                        detector_field, amplitude, pod.mask
                    )

                    updated = self.object_update(
                        current,
                        pod.bw(projected),
                        cdi_support(self.ptycho, pod),
                    )
                    pod.object = updated

                    norm = np.linalg.norm(current)
                    change = np.linalg.norm(updated - current) / norm if norm > 0 else 0.0

                    error_dct[name] = np.array([amplitude_error, 0.0, change])

            self.curiter += 1

        return error_dct

    def object_update(self, current, candidate, support):
        """Return the new object from the current object and the candidate."""
        raise NotImplementedError

    def engine_finalize(self):
        for pod in self.pods.values():
            pod.object = remove_global_phase(
                pod.object,
                cdi_support(self.ptycho, pod),
            )
