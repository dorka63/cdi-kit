"""Real- and reciprocal-space constraints for CDI reconstruction."""

from __future__ import annotations

import numpy as np


def masked_modulus_projection(
    wavefield: np.ndarray,
    measured_amplitude: np.ndarray,
    valid_mask: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Project a detector-plane field onto measured amplitudes.

    The phase of ``wavefield`` is retained at valid detector pixels, while its
    amplitude is replaced by ``measured_amplitude``. Pixels outside
    ``valid_mask`` are left unchanged; they represent unmeasured detector
    regions such as beamstops, detector gaps, dead pixels, or saturated pixels.

    Parameters
    ----------
    wavefield
        Complex-valued detector-plane field.
    measured_amplitude
        Non-negative measured amplitude, normally ``sqrt(intensity)``.
    valid_mask
        Boolean mask: ``True`` for pixels constrained by experimental data and
        ``False`` for pixels without a usable measurement.
    epsilon
        Threshold used to define a deterministic phase where the current
        wavefield amplitude is zero.

    Returns
    -------
    numpy.ndarray
        A complex array with the same shape as ``wavefield``.
    """
    wavefield = np.asarray(wavefield)
    measured_amplitude = np.asarray(measured_amplitude)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if wavefield.shape != measured_amplitude.shape:
        raise ValueError(
            "wavefield and measured_amplitude must have identical shapes; "
            f"got {wavefield.shape} and {measured_amplitude.shape}."
        )

    if wavefield.shape != valid_mask.shape:
        raise ValueError(
            "wavefield and valid_mask must have identical shapes; "
            f"got {wavefield.shape} and {valid_mask.shape}."
        )

    if not np.iscomplexobj(wavefield):
        raise TypeError("wavefield must be a complex-valued array.")

    if not np.all(np.isfinite(measured_amplitude)):
        raise ValueError("measured_amplitude must contain only finite values.")

    if np.any(measured_amplitude < 0):
        raise ValueError("measured_amplitude must be non-negative.")

    current_amplitude = np.abs(wavefield)

    phase_factor = np.ones_like(wavefield, dtype=np.complex128)
    nonzero = current_amplitude > epsilon
    phase_factor[nonzero] = (
        wavefield[nonzero] / current_amplitude[nonzero]
    )

    projected = np.array(wavefield, dtype=np.complex128, copy=True)
    projected[valid_mask] = (
        measured_amplitude[valid_mask] * phase_factor[valid_mask]
    )

    return projected