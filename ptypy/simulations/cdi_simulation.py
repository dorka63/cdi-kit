"""Synthetic complex objects for CDI tests and tutorials.

Diffraction intensities are not computed here: use the PtyPy propagator,
for example ``CDIGeometry.forward``, so that simulation and reconstruction
share the same physics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ptypy import io
from ptypy.custom.cdi_common import _validate_shape, circular_support


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

    return object_field.astype(np.complex128), support