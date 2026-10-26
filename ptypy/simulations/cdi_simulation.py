"""Synthetic complex objects for CDI tests and tutorials.

Diffraction intensities are not computed here: use the PtyPy propagator,
for example ``CDIGeometry.forward``, so that simulation and reconstruction
share the same physics.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from ptypy import io
from ptypy.custom.cdi_common import _validate_shape, circular_support, remove_global_phase


Array = np.ndarray


def _to_grayscale(image: Array) -> Array:
    """Convert a grayscale, RGB, or RGBA image to floating grayscale."""
    image = np.asarray(image)

    if image.ndim == 2:
        grayscale = image.astype(np.float64)
    elif image.ndim == 3 and image.shape[-1] in {3, 4}:
        rgb = image[..., :3].astype(np.float64)
        grayscale = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    else:
        raise ValueError(
            "amplitude image must be grayscale, RGB, or RGBA; "
            f"got shape {image.shape}."
        )

    if not np.all(np.isfinite(grayscale)):
        raise ValueError("amplitude image must contain finite values.")

    return grayscale


def _normalize_image(image: Array) -> Array:
    """Normalize a finite image to the closed interval [0, 1]."""
    image = np.asarray(image, dtype=np.float64)
    image_min = float(image.min())
    image_max = float(image.max())

    if image_max <= image_min:
        raise ValueError(
            "amplitude image must contain more than one intensity value."
        )

    return (image - image_min) / (image_max - image_min)


def amplitude_from_image(
    image_path: str | Path,
    *,
    invert: bool = False,
    amplitude_floor: float = 0.0,
    amplitude_ceiling: float = 1.0,
    binary_threshold: float | None = 0.5,
) -> Array:
    """Read an image amplitude map without changing its sampling geometry.

    The image is read through ``ptypy.io.image_read``, converted to grayscale
    and normalized to [0, 1]. It is never cropped, resized, padded, or
    translated. With ``binary_threshold`` set, pixels below the threshold map
    to ``amplitude_floor`` and the rest to ``amplitude_ceiling``.
    """
    if amplitude_floor < 0.0:
        raise ValueError("amplitude_floor must be non-negative.")

    if amplitude_ceiling < amplitude_floor:
        raise ValueError(
            "amplitude_ceiling must be greater than or equal to amplitude_floor."
        )

    if binary_threshold is not None and not 0.0 < binary_threshold < 1.0:
        raise ValueError(
            "binary_threshold must lie strictly between 0 and 1 or be None."
        )

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"amplitude image does not exist: {image_path}")

    image, _metadata = io.image_read(str(image_path))
    normalized = _normalize_image(_to_grayscale(image))

    if invert:
        normalized = 1.0 - normalized

    if binary_threshold is not None:
        normalized = (normalized >= binary_threshold).astype(np.float64)

    return amplitude_floor + (amplitude_ceiling - amplitude_floor) * normalized


def _centred_coordinates(shape: tuple[int, int]) -> tuple[Array, Array]:
    """Return normalized centred coordinates ``(y, x)``."""
    rows, columns = shape
    row, column = np.indices((rows, columns), dtype=float)
    y = (row - (rows - 1) / 2.0) / rows
    x = (column - (columns - 1) / 2.0) / columns
    return y, x


def _gaussian_feature_amplitude(shape: tuple[int, int]) -> Array:
    """Return the asymmetric Gaussian-feature amplitude model."""
    y, x = _centred_coordinates(shape)
    return (
        0.25
        + 0.55 * np.exp(-((x + 0.12) ** 2 + (y - 0.08) ** 2) / 0.008)
        + 0.35 * np.exp(-((x - 0.15) ** 2 + (y + 0.13) ** 2) / 0.003)
    )


def _default_phase(shape: tuple[int, int]) -> Array:
    """Return the smooth asymmetric phase model."""
    y, x = _centred_coordinates(shape)
    return (
        1.2 * np.exp(-((x + 0.08) ** 2 + (y + 0.10) ** 2) / 0.012)
        - 0.8 * np.exp(-((x - 0.17) ** 2 + (y - 0.15) ** 2) / 0.006)
        + 0.5 * x
    )


def make_complex_test_object(
    shape: tuple[int, int] = (128, 128),
    *,
    amplitude_model: str = "gaussian_features",
    amplitude_image_path: str | Path | None = None,
    image_invert: bool = False,
    image_binary_threshold: float | None = 0.5,
    amplitude_floor: float = 0.0,
    amplitude_ceiling: float = 1.0,
    uniform_amplitude: float = 1.0,
    support_radius_fraction: float = 0.2,
) -> tuple[Array, Array]:
    """Create a deterministic complex CDI phantom and its circular support.

    ``amplitude_model`` is ``"gaussian_features"``, ``"uniform"``, or
    ``"image"``. For ``"image"`` the image only defines the amplitude, its
    shape must equal ``shape`` exactly, and it is never resized. The phase is
    always the smooth model from ``_default_phase``.
    """
    rows, columns = _validate_shape(shape)

    if not 0.0 < support_radius_fraction <= 0.5:
        raise ValueError(
            "support_radius_fraction must lie in the interval (0, 0.5]."
        )

    support = circular_support(
        shape=(rows, columns),
        radius_px=min(rows, columns) * support_radius_fraction,
    )

    if amplitude_model == "gaussian_features":
        amplitude = _gaussian_feature_amplitude((rows, columns))

    elif amplitude_model == "uniform":
        if uniform_amplitude < 0.0:
            raise ValueError("uniform_amplitude must be non-negative.")

        amplitude = np.full((rows, columns), float(uniform_amplitude))

    elif amplitude_model == "image":
        if amplitude_image_path is None:
            raise ValueError(
                "amplitude_image_path is required when amplitude_model='image'."
            )

        amplitude = amplitude_from_image(
            amplitude_image_path,
            invert=image_invert,
            amplitude_floor=amplitude_floor,
            amplitude_ceiling=amplitude_ceiling,
            binary_threshold=image_binary_threshold,
        )

        if amplitude.shape != (rows, columns):
            raise ValueError(
                "shape must match the image shape exactly; got "
                f"shape={(rows, columns)}, image shape={amplitude.shape}."
            )

    else:
        raise ValueError(
            "amplitude_model must be 'gaussian_features', 'uniform', or 'image'."
        )

    object_field = support * amplitude * np.exp(1j * _default_phase((rows, columns)))

    object_field = object_field.astype(np.complex128)
    if np.any(object_field):
        object_field = remove_global_phase(object_field, support)
    return object_field, support

def shift_diffraction_pattern(intensity, shift_px=(7, -5), *, valid_mask=None):
    """Controlled reversible integer registration test using periodic boundaries.

    This is not a finite-detector translation model. No interpolation or
    object translation is performed; a fixed detector gap is applied later.
    """
    intensity = np.asarray(intensity)
    if intensity.ndim != 2 or any(size == 0 for size in intensity.shape):
        raise ValueError("intensity must be a nonempty two-dimensional array.")
    if np.iscomplexobj(intensity):
        raise ValueError("intensity must be real-valued.")
    if not np.isfinite(intensity).all() or np.any(intensity < 0):
        raise ValueError("intensity must be finite and nonnegative.")
    shift = np.asarray(shift_px, dtype=float)
    if shift.shape != (2,) or not np.isfinite(shift).all():
        raise ValueError("shift_px must contain two finite integer values.")
    if not np.array_equal(shift, np.rint(shift)):
        raise ValueError("Fractional detector shifts are not supported.")
    if np.any(np.abs(shift) > np.iinfo(np.intp).max // 2):
        raise ValueError("shift_px is outside the supported integer range.")
    shift = tuple(int(value) for value in shift)
    mask = np.ones(intensity.shape, bool) if valid_mask is None else np.asarray(valid_mask, bool)
    if mask.shape != intensity.shape:
        raise ValueError("valid_mask and intensity must have identical shapes.")
    return np.roll(intensity, shift, axis=(0, 1)), np.roll(mask, shift, axis=(0, 1))


def make_physical_image_object(
    geometry,
    image_path: str | Path,
    *,
    diameter_m: float,
    image_invert: bool = False,
    image_binary_threshold: float | None = 0.5,
    phase_model: str = "smooth",
) -> tuple[Array, Array]:
    """Place an image phantom on an existing physical PtyPy object grid.

    diameter_m is the largest side of the nonzero foreground bounding box;
    for a Siemens star this is its outer diameter. Pixel-cell extents are
    used, so the realized diameter is quantized to the object-grid sampling.
    Aspect ratio is retained in physical coordinates, including anisotropic
    sampling. Only this explicit physical-size API resamples in memory;
    amplitude_from_image and make_complex_test_object retain their APIs.
    The source image file is never modified. The returned mask is the true
    nonzero support, not a reconstruction support proposal.
    """
    from scipy.ndimage import zoom

    diameter_m = float(diameter_m)
    if not np.isfinite(diameter_m) or diameter_m <= 0:
        raise ValueError("diameter_m must be positive and finite.")
    shape = _validate_shape(tuple(geometry.shape))
    pitch = np.asarray(geometry.object_psize_m, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("geometry.object_psize_m must contain two positive finite values.")
    if phase_model not in ("smooth", "zero"):
        raise ValueError("phase_model must be 'smooth' or 'zero'.")
    image = amplitude_from_image(image_path, invert=image_invert,
                                 binary_threshold=image_binary_threshold)
    row, column = np.nonzero(image > 0)
    if row.size == 0:
        raise ValueError("Image foreground is empty; check image_invert.")
    pattern = image[row.min():row.max()+1, column.min():column.max()+1]
    physical_extent = np.asarray(pattern.shape, float) / max(pattern.shape) * diameter_m
    target = np.rint(physical_extent / pitch).astype(int)
    if np.any(target < 1) or np.any(target > np.asarray(shape)):
        raise ValueError("Requested physical object does not fit in the geometry or is below sampling.")
    sampled = zoom(pattern, target / np.asarray(pattern.shape), order=1)
    if sampled.shape != tuple(target):
        raise RuntimeError("Unexpected resampling shape.")
    amplitude = np.zeros(shape, dtype=np.float64)
    start = np.asarray(shape)//2 - target//2
    amplitude[start[0]:start[0]+target[0], start[1]:start[1]+target[1]] = sampled
    true_support = amplitude > 0
    phase = _default_phase(shape) if phase_model == "smooth" else np.zeros(shape)
    obj = (amplitude * np.exp(1j*phase)).astype(np.complex128)
    return remove_global_phase(obj, true_support), true_support
