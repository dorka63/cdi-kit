"""Small far-field CDI simulation utilities for tests and tutorials."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ptypy import io


Array = np.ndarray


def fft2_centered(field: Array) -> Array:
    """Return a centered unitary two-dimensional Fourier transform."""
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


def ifft2_centered(spectrum: Array) -> Array:
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


def farfield_intensity(object_field: Array) -> Array:
    """Simulate centered far-field intensity from a complex object field."""
    detector_field = fft2_centered(object_field)
    return np.abs(detector_field) ** 2


def circular_support(
    shape: tuple[int, int],
    radius_px: float,
    center_px: tuple[float, float] | None = None,
) -> Array:
    """Create a boolean circular support mask in array coordinates."""
    if len(shape) != 2:
        raise ValueError("shape must contain exactly two dimensions.")

    rows, columns = int(shape[0]), int(shape[1])

    if rows <= 0 or columns <= 0:
        raise ValueError("shape dimensions must be positive.")

    if radius_px <= 0:
        raise ValueError("radius_px must be positive.")

    if center_px is None:
        center_row = (rows - 1) / 2.0
        center_column = (columns - 1) / 2.0
    else:
        if len(center_px) != 2:
            raise ValueError(
                "center_px must contain exactly two coordinates."
            )

        center_row = float(center_px[0])
        center_column = float(center_px[1])

    row, column = np.indices((rows, columns), dtype=float)

    return (
        (row - center_row) ** 2
        + (column - center_column) ** 2
        <= float(radius_px) ** 2
    )


def _validate_shape(
    shape: tuple[int, int],
) -> tuple[int, int]:
    """Validate and return a two-dimensional array shape."""
    if len(shape) != 2:
        raise ValueError("shape must contain exactly two dimensions.")

    rows, columns = int(shape[0]), int(shape[1])

    if rows <= 0 or columns <= 0:
        raise ValueError("shape dimensions must be positive.")

    return rows, columns


def _to_grayscale(image: Array) -> Array:
    """Convert a grayscale, RGB, or RGBA image to floating grayscale."""
    image = np.asarray(image)

    if image.ndim == 2:
        grayscale = image.astype(np.float64)

    elif image.ndim == 3 and image.shape[-1] in {3, 4}:
        rgb = image[..., :3].astype(np.float64)

        grayscale = (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        )

    else:
        raise ValueError(
            "amplitude image must be grayscale, RGB, or RGBA; "
            f"got shape {image.shape}."
        )

    if not np.all(np.isfinite(grayscale)):
        raise ValueError(
            "amplitude image must contain finite values."
        )

    return grayscale


def _normalize_image(image: Array) -> Array:
    """Normalize a finite image to the closed interval [0, 1]."""
    image = np.asarray(image, dtype=np.float64)

    image_min = float(image.min())
    image_max = float(image.max())

    if not np.isfinite(image_min) or not np.isfinite(image_max):
        raise ValueError(
            "amplitude image must have finite extrema."
        )

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

    The image is read through :func:`ptypy.io.image_read`, converted to
    grayscale and normalized to [0, 1]. It is never cropped, resized, padded,
    translated, or otherwise geometrically transformed.

    When ``binary_threshold`` is a number, the normalized grayscale image is
    converted to a binary amplitude map before the amplitude-range mapping:

    - normalized image < binary_threshold -> amplitude_floor;
    - normalized image >= binary_threshold -> amplitude_ceiling.

    Parameters
    ----------
    image_path
        Path to an image readable by ``ptypy.io.image_read``.
    invert
        If True, replace normalized grayscale value ``g`` with ``1 - g``.
    amplitude_floor
        Output amplitude assigned to black pixels.
    amplitude_ceiling
        Output amplitude assigned to white pixels.
    binary_threshold
        Threshold in the open interval (0, 1). Set it to ``None`` to retain
        a continuous grayscale amplitude map.

    Returns
    -------
    numpy.ndarray
        Real-valued amplitude map with exactly the image's row-column shape.
    """
    if amplitude_floor < 0.0:
        raise ValueError("amplitude_floor must be non-negative.")

    if amplitude_ceiling < amplitude_floor:
        raise ValueError(
            "amplitude_ceiling must be greater than or equal to "
            "amplitude_floor."
        )

    if binary_threshold is not None:
        if not 0.0 < binary_threshold < 1.0:
            raise ValueError(
                "binary_threshold must lie strictly between 0 and 1 "
                "or be None."
            )

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"amplitude image does not exist: {image_path}"
        )

    image, _metadata = io.image_read(str(image_path))

    grayscale = _to_grayscale(image)
    normalized = _normalize_image(grayscale)

    if invert:
        normalized = 1.0 - normalized

    if binary_threshold is not None:
        normalized = (
            normalized >= binary_threshold
        ).astype(np.float64)

    return (
        amplitude_floor
        + (amplitude_ceiling - amplitude_floor) * normalized
    )


def _gaussian_feature_amplitude(
    shape: tuple[int, int],
) -> Array:
    """Return the original asymmetric Gaussian-feature amplitude model."""
    rows, columns = _validate_shape(shape)

    row, column = np.indices((rows, columns), dtype=float)

    y = (row - (rows - 1) / 2.0) / rows
    x = (column - (columns - 1) / 2.0) / columns

    return (
        0.25
        + 0.55 * np.exp(
            -((x + 0.12) ** 2 + (y - 0.08) ** 2) / 0.008
        )
        + 0.35 * np.exp(
            -((x - 0.15) ** 2 + (y + 0.13) ** 2) / 0.003
        )
    )


def _uniform_amplitude(
    shape: tuple[int, int],
    value: float,
) -> Array:
    """Return a constant-amplitude model."""
    rows, columns = _validate_shape(shape)

    if value < 0.0:
        raise ValueError("uniform amplitude must be non-negative.")

    return np.full(
        (rows, columns),
        fill_value=float(value),
        dtype=np.float64,
    )


def _default_phase(
    shape: tuple[int, int],
) -> Array:
    """Return the original smooth asymmetric phase model."""
    rows, columns = _validate_shape(shape)

    row, column = np.indices((rows, columns), dtype=float)

    y = (row - (rows - 1) / 2.0) / rows
    x = (column - (columns - 1) / 2.0) / columns

    return (
        1.2 * np.exp(
            -((x + 0.08) ** 2 + (y + 0.10) ** 2) / 0.012
        )
        - 0.8 * np.exp(
            -((x - 0.17) ** 2 + (y - 0.15) ** 2) / 0.006
        )
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
    """Create a deterministic complex CDI phantom and a circular support.

    For ``amplitude_model="image"``, the external image is used only for the
    amplitude. The smooth phase returned by :func:`_default_phase` remains
    unchanged. The image grid is required to match ``shape`` exactly and is
    never resized.

    Parameters
    ----------
    shape
        Shape of the object field. With ``amplitude_model="image"``, it must
        exactly equal the image row-column shape.
    amplitude_model
        ``"gaussian_features"``, ``"uniform"``, or ``"image"``.
    amplitude_image_path
        Image path used for ``amplitude_model="image"``.
    image_invert
        Invert image brightness before thresholding/mapping amplitude.
    image_binary_threshold
        Binary threshold for image amplitude. Set to ``None`` to retain a
        continuous grayscale amplitude.
    amplitude_floor, amplitude_ceiling
        Output range used for the image-amplitude model.
    uniform_amplitude
        Constant amplitude used by ``amplitude_model="uniform"``.
    support_radius_fraction
        Circular-support radius divided by the smaller array dimension.

    Returns
    -------
    object_field, support
        Complex object field and boolean circular support mask.
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
        amplitude = _uniform_amplitude(
            shape=(rows, columns),
            value=uniform_amplitude,
        )

    elif amplitude_model == "image":
        if amplitude_image_path is None:
            raise ValueError(
                "amplitude_image_path is required when "
                "amplitude_model='image'."
            )

        amplitude = amplitude_from_image(
            image_path=amplitude_image_path,
            invert=image_invert,
            amplitude_floor=amplitude_floor,
            amplitude_ceiling=amplitude_ceiling,
            binary_threshold=image_binary_threshold,
        )

        if amplitude.shape != (rows, columns):
            raise ValueError(
                "shape must match the image shape exactly for "
                "amplitude_model='image'; got "
                f"shape={(rows, columns)}, image shape={amplitude.shape}. "
                "The supplied image is intentionally not resized."
            )

    else:
        raise ValueError(
            "amplitude_model must be one of: "
            "'gaussian_features', 'uniform', or 'image'."
        )

    phase = _default_phase((rows, columns))

    object_field = support * amplitude * np.exp(1j * phase)

    return object_field.astype(np.complex128), support