"""Small far-field CDI simulation utilities for testing and examples."""

from __future__ import annotations

import numpy as np


def fft2_centered(field: np.ndarray) -> np.ndarray:
    """Return a centered 2D Fourier transform of a complex field."""
    field = np.asarray(field)

    if field.ndim != 2:
        raise ValueError(
            f"field must be two-dimensional; got ndim={field.ndim}."
        )

    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(field),
            norm="ortho",
        )
    )


def ifft2_centered(spectrum: np.ndarray) -> np.ndarray:
    """Return the inverse of :func:`fft2_centered`."""
    spectrum = np.asarray(spectrum)

    if spectrum.ndim != 2:
        raise ValueError(
            f"spectrum must be two-dimensional; got ndim={spectrum.ndim}."
        )

    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(spectrum),
            norm="ortho",
        )
    )


def farfield_intensity(object_field: np.ndarray) -> np.ndarray:
    """Simulate a centered far-field intensity pattern from an object field."""
    detector_field = fft2_centered(object_field)
    return np.abs(detector_field) ** 2


def circular_support(
    shape: tuple[int, int],
    radius_px: float,
    center_px: tuple[float, float] | None = None,
) -> np.ndarray:
    """Create a boolean circular support mask in array coordinates."""
    if len(shape) != 2:
        raise ValueError("shape must contain exactly two dimensions.")

    height, width = shape

    if height <= 0 or width <= 0:
        raise ValueError("shape dimensions must be positive.")

    if radius_px <= 0:
        raise ValueError("radius_px must be positive.")

    if center_px is None:
        center_y = (height - 1) / 2.0
        center_x = (width - 1) / 2.0
    else:
        center_y, center_x = center_px

    y, x = np.indices(shape)
    return (y - center_y) ** 2 + (x - center_x) ** 2 <= radius_px**2


def make_complex_test_object(
    shape: tuple[int, int] = (128, 128),
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic complex phantom and its known support.

    The phantom contains both amplitude and phase structure. It is asymmetric
    by design, which makes accidental axis swaps, flips, and shift errors
    easier to detect in tests.
    """
    support = circular_support(shape, radius_px=min(shape) * 0.28)

    height, width = shape
    y, x = np.indices(shape, dtype=float)
    y = (y - (height - 1) / 2.0) / height
    x = (x - (width - 1) / 2.0) / width

    amplitude = (
        0.25
        + 0.55 * np.exp(-((x + 0.12) ** 2 + (y - 0.08) ** 2) / 0.008)
        + 0.35 * np.exp(-((x - 0.15) ** 2 + (y + 0.13) ** 2) / 0.003)
    )

    phase = (
        1.2 * np.exp(-((x + 0.08) ** 2 + (y + 0.10) ** 2) / 0.012)
        - 0.8 * np.exp(-((x - 0.17) ** 2 + (y - 0.15) ** 2) / 0.006)
        + 0.5 * x
    )

    object_field = support * amplitude * np.exp(1j * phase)
    return object_field.astype(np.complex128), support