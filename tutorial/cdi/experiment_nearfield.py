"""Reconstruct experimental near-field data using PtyPy CDI engines.

Run from the repository root:
    python tutorial/cdi/experiment_nearfield.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter

from ptypy import io, utils as u
from ptypy.core import Ptycho
from ptypy.custom.cdi_common import (
    cdi_support,
    set_cdi_support,
    rectangular_support,
)
import ptypy.custom.cdi_er
import ptypy.custom.cdi_hio


IMAGE_PATH = Path(__file__).resolve().parent / "data" / "1.tif"
OUTPUT = Path("output/experiment_nearfield")

Z_UM = 6e5
LAMBDA_UM = 0.6328
PIXEL_SIZE_UM = (6.0, 6.0)

BETAS = (0.7, 0.4, 0.1)
HIO_ITERATIONS = 30
ER_ITERATIONS = 10
INNER_REPEATS = 3
OUTER_CYCLES = 3
RESET_PHASE = True

SIGMA_CLIP = 3.0
SIGMA_CLIP_MAX_ITER = 5
STATISTICS_MASK = None

SUPPORT_THRESHOLD = 0.04
SUPPORT_BLUR_SIGMA = 1.0
SUPPORT_CLOSING_ITERATIONS = 0

SHRINKWRAP_SIGMA = 2.0
SHRINKWRAP_THRESHOLD = 0.1
SHRINKWRAP_CLOSING_ITERATIONS = 0


def main():
    intensity, _ = io.image_read(str(IMAGE_PATH))
    intensity = np.asarray(intensity, dtype=np.float64)

    if intensity.shape != (1030, 1288):
        raise ValueError(
            f"Expected image shape (1030, 1288), got {intensity.shape}."
        )

    if min(
        HIO_ITERATIONS,
        ER_ITERATIONS,
        INNER_REPEATS,
        OUTER_CYCLES,
    ) < 1:
        raise ValueError(
            "Iteration counts and repeat counts must be positive."
        )

    # Use a central square ROI before constructing the initial support.
    size = min(intensity.shape)
    row_start = (intensity.shape[0] - size) // 2
    column_start = (intensity.shape[1] - size) // 2

    roi = (
        slice(row_start, row_start + size),
        slice(column_start, column_start + size),
    )

    statistics_mask = STATISTICS_MASK

    if statistics_mask is not None:
        statistics_mask = np.asarray(statistics_mask)

        if statistics_mask.shape != intensity.shape:
            raise ValueError(
                "STATISTICS_MASK must match the original TIFF shape."
            )

        if statistics_mask.dtype != np.dtype(bool):
            raise TypeError("STATISTICS_MASK must be a boolean array.")

        statistics_mask = statistics_mask[roi].copy()

    intensity = intensity[roi].copy()
    support = rectangular_support( shape=intensity.shape, oversampling=2.0,)
    p = u.Param()
    p.verbose_level = "info"
    p.data_type = "double"

    p.io = u.Param(
        rfile=None,
        autosave=u.Param(active=False),
        autoplot=u.Param(active=False),
        interaction=u.Param(active=False),
    )

    p.scans = u.Param()
    p.scans.cdi = u.Param(
        name="Vanilla",
        propagation="nearfield",
        ffttype="scipy",
    )

    p.scans.cdi.data = u.Param(
        name="CDIScan",
        intensity=intensity,
        energy=1.23984193e-9 / (LAMBDA_UM * 1e-6),
        distance=Z_UM * 1e-6,
        psize=tuple(value * 1e-6 for value in PIXEL_SIZE_UM),
        center=None,
        autocenter=True,
        autocenter_method="mass",
        autocenter_initial_center=None,
        autocenter_search_radius=14.0,
        autocenter_tolerance=1e-4,
        autocenter_min_overlap=0.5,
        rebin=1,
        orientation=None,
        support=u.Param(
            kind="array",
            array=support,
        ),

        denoising=u.Param(
            active=True,
            name="sigma_clip",
            sigma=SIGMA_CLIP,
            max_iter=SIGMA_CLIP_MAX_ITER,
            statistics_mask=statistics_mask,
            verbose=True,
        ),
    )

    p.engines = u.Param()

    reconstruction = Ptycho(p, level=2)
    pod = next(iter(reconstruction.pods.values()))
    initial_support = cdi_support(reconstruction, pod).copy()

    if initial_support.shape != pod.object.shape:
        raise ValueError(
            "Support and object shapes do not match: "
            f"{initial_support.shape} and {pod.object.shape}."
        )

    if not np.any(pod.diff > 0):
        raise ValueError("Background subtraction removed all signal.")

    for cycle in range(OUTER_CYCLES):
        for _ in range(INNER_REPEATS):
            for beta in BETAS:
                reconstruction.run(
                    epars=u.Param(
                        name="CDIHIO",
                        numiter=HIO_ITERATIONS,
                        numiter_contiguous=1,
                        beta=beta,
                    )
                )

                reconstruction.run(
                    epars=u.Param(
                        name="CDIER",
                        numiter=ER_ITERATIONS,
                        numiter_contiguous=1,
                    )
                )

        amplitude = gaussian_filter(
            np.abs(pod.object),
            SHRINKWRAP_SIGMA,
        )

        if not np.isfinite(amplitude).all() or amplitude.max() <= 0:
            raise ValueError(
                "Cannot update support from a zero/nonfinite object."
            )

        support = (
            amplitude >= SHRINKWRAP_THRESHOLD * amplitude.max()
        )

        if SHRINKWRAP_CLOSING_ITERATIONS:
            support = binary_closing(
                support,
                iterations=SHRINKWRAP_CLOSING_ITERATIONS,
            )

        set_cdi_support(reconstruction, pod, support)

        if RESET_PHASE and cycle + 1 < OUTER_CYCLES:
            pod.object = np.abs(pod.object).astype(
                reconstruction.CType
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    obj = pod.object.copy()
    support = cdi_support(reconstruction, pod).copy()

    np.savez_compressed(
        OUTPUT / "reconstruction.npz",
        object_field=obj,
        support=support,
        initial_support=initial_support,
        processed_intensity=pod.diff.copy(),
    )

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))

    for axis, image, title, cmap in zip(
        axes,
        (np.abs(obj), np.angle(obj), support),
        ("Object amplitude", "Object phase", "Support"),
        ("viridis", "twilight", "gray"),
    ):
        artist = axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        figure.colorbar(artist, ax=axis, shrink=0.8)

    figure.tight_layout()
    figure.savefig(OUTPUT / "reconstruction.png", dpi=150)
    plt.close(figure)

    reconstruction.finalize()


if __name__ == "__main__":
    main()