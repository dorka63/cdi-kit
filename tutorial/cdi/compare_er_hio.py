"""Compare PtyPy CDI ER, HIO, and RAAR on a shifted Siemens-star pattern."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.custom.cdi_common import (
    CDIGeometry,
    cdi_support,
)
import ptypy.custom.cdi_er
import ptypy.custom.cdi_hio
import ptypy.custom.cdi_raar
from ptypy.simulations.cdi_simulation import (
    amplitude_from_image,
    make_complex_test_object,
    shift_diffraction_pattern,
)


Array = np.ndarray

IMAGE_PATH = Path(__file__).resolve().parent / "data" / "star.png"
OUTPUT_DIRECTORY = Path("output") / "cdi_er_hio_raar_example"

ENERGY_KEV = 8.0
DISTANCE_M = 1.0
DETECTOR_PSIZE_M = 55e-6

TOTAL_ITERATIONS = 360
HIO_BETA = 0.7
RAAR_BETA = 0.9

DETECTOR_SHIFT_PX = (7, -5)
AUTOCENTER = True
AUTOCENTER_METHOD = "reflections"
AUTOCENTER_INITIAL_CENTER = None
AUTOCENTER_SEARCH_RADIUS_PX = 12.0
AUTOCENTER_TOLERANCE = 1e-4
AUTOCENTER_MIN_OVERLAP = 0.5
MANUAL_CENTER_PX = None


@dataclass
class ReconstructionResult:
    """Final object, support, and error histories for one reconstruction."""

    name: str
    object_field: Array
    support: Array
    amplitude_errors: list[float]
    object_changes: list[float]


def align_global_phase(
    reference: Array,
    reconstruction: Array,
    mask: Array,
) -> Array:
    """Align the reconstruction's global phase with the reference."""
    overlap = np.vdot(reference[mask], reconstruction[mask])

    if np.abs(overlap) == 0.0:
        return reconstruction.copy()

    return reconstruction * np.exp(-1j * np.angle(overlap))


def complex_nrmse(
    reference: Array,
    reconstruction: Array,
    mask: Array,
) -> float:
    """Return complex NRMSE inside mask after global-phase alignment."""
    aligned = align_global_phase(reference, reconstruction, mask)
    numerator = np.linalg.norm((aligned - reference)[mask])
    denominator = np.linalg.norm(reference[mask])

    if denominator == 0.0:
        raise ValueError("Reference norm inside the mask is zero.")

    return float(numerator / denominator)


def show(axis, figure, image, title, colormap) -> None:
    """Draw one image panel with a title and colour bar."""
    artist = axis.imshow(image, cmap=colormap)
    axis.set_title(title, fontsize=10)
    axis.set_axis_off()
    figure.colorbar(artist, ax=axis, shrink=0.78)


def save_reconstruction_figure(
    output_path: Path,
    true_object: Array,
    known_support: Array,
    intensity: Array,
    results: list[ReconstructionResult],
) -> None:
    """Save reference data and final reconstructions."""
    figure, axes = plt.subplots(
        nrows=1 + len(results),
        ncols=4,
        figsize=(16, 4 * (1 + len(results))),
        constrained_layout=True,
    )

    show(
        axes[0, 0],
        figure,
        np.abs(true_object),
        "True amplitude",
        "viridis",
    )
    show(
        axes[0, 1],
        figure,
        np.ma.masked_where(~known_support, np.angle(true_object)),
        "True phase",
        "twilight",
    )
    show(
        axes[0, 2],
        figure,
        np.log10(intensity + 1e-12),
        "log10 shifted far-field intensity",
        "magma",
    )
    show(
        axes[0, 3],
        figure,
        known_support.astype(float),
        "Common known support",
        "gray",
    )

    for row, result in enumerate(results, start=1):
        aligned = align_global_phase(
            true_object,
            result.object_field,
            known_support,
        )

        show(
            axes[row, 0],
            figure,
            np.abs(aligned),
            f"{result.name}: amplitude",
            "viridis",
        )
        show(
            axes[row, 1],
            figure,
            np.ma.masked_where(~result.support, np.angle(aligned)),
            f"{result.name}: phase",
            "twilight",
        )
        show(
            axes[row, 2],
            figure,
            np.abs(aligned - true_object),
            f"{result.name}: |complex error|",
            "magma",
        )
        show(
            axes[row, 3],
            figure,
            result.support.astype(float),
            f"{result.name}: support",
            "gray",
        )

    figure.savefig(str(output_path.resolve()), dpi=150)
    plt.close(figure)


def save_error_plot(
    output_path: Path,
    results: list[ReconstructionResult],
) -> None:
    """Save amplitude residual and relative object-change histories."""
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 4.8),
        constrained_layout=True,
    )

    for result in results:
        iterations = np.arange(1, len(result.amplitude_errors) + 1)

        axes[0].semilogy(
            iterations,
            np.maximum(result.amplitude_errors, 1e-16),
            label=result.name,
        )
        axes[1].semilogy(
            iterations,
            np.maximum(result.object_changes, 1e-16),
            label=result.name,
        )

    axes[0].set_title("Masked amplitude residual")
    axes[1].set_title("Relative object change")

    for axis in axes:
        axis.set_xlabel("Iteration")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.savefig(str(output_path.resolve()), dpi=150)
    plt.close(figure)


def make_parameters(
    intensity: Array,
    support: Array,
    engines: u.Param,
    *,
    valid_mask: Array | None = None,
) -> u.Param:
    """Create a PtyPy configuration for single-frame CDI reconstruction."""
    if not AUTOCENTER and MANUAL_CENTER_PX is None:
        raise ValueError(
            "Set MANUAL_CENTER_PX when AUTOCENTER is False."
        )

    parameters = u.Param()
    parameters.verbose_level = "info"
    parameters.data_type = "double"

    parameters.io = u.Param(
        rfile=None,
        autosave=u.Param(active=False),
        autoplot=u.Param(active=False),
        interaction=u.Param(active=False),
    )

    parameters.scans = u.Param()
    parameters.scans.cdi = u.Param(
        name="Vanilla",
        propagation="farfield",
        ffttype="numpy",
    )

    parameters.scans.cdi.data = u.Param(
        name="CDIScan",
        intensity=intensity,
        mask=valid_mask,
        center=None if AUTOCENTER else MANUAL_CENTER_PX,
        autocenter=AUTOCENTER,
        autocenter_method=AUTOCENTER_METHOD,
        autocenter_initial_center=(
            tuple(np.asarray(intensity.shape, dtype=float) / 2.0)
            if AUTOCENTER_INITIAL_CENTER is None
            else AUTOCENTER_INITIAL_CENTER
        ),
        autocenter_search_radius=AUTOCENTER_SEARCH_RADIUS_PX,
        autocenter_tolerance=AUTOCENTER_TOLERANCE,
        autocenter_min_overlap=AUTOCENTER_MIN_OVERLAP,
        orientation=None,
        rebin=1,
        save=None,
        energy=ENERGY_KEV,
        distance=DISTANCE_M,
        psize=DETECTOR_PSIZE_M,
        support=u.Param(kind="array", array=support),
    )

    parameters.engines = engines
    return parameters


def result_from_ptycho(
    name: str,
    reconstruction: Ptycho,
) -> ReconstructionResult:
    """Extract the final CDI object, support, and histories."""
    pod = next(iter(reconstruction.pods.values()))

    return ReconstructionResult(
        name=name,
        object_field=pod.object.copy(),
        support=cdi_support(reconstruction, pod).copy(),
        amplitude_errors=[
            float(info["error"][0])
            for info in reconstruction.runtime.iter_info
        ],
        object_changes=[
            float(info["error"][2])
            for info in reconstruction.runtime.iter_info
        ],
    )


def report_centering(
    name: str,
    reconstruction: Ptycho,
    reference_center: Array,
    expected_center: Array,
    input_intensity: Array,
    input_mask: Array,
) -> Array:
    """Report the center chosen by the reconstruction's CDIScan."""
    model = reconstruction.model.scans["cdi"]
    measured_center = np.asarray(
        model.ptyscan.meta.center,
        dtype=float,
    )
    geometry_center = np.asarray(
        model.geometries[0].p.center,
        dtype=float,
    )
    pod = next(iter(reconstruction.pods.values()))

    np.testing.assert_array_equal(pod.diff, input_intensity)
    np.testing.assert_array_equal(pod.mask, input_mask)
    np.testing.assert_allclose(
        geometry_center,
        measured_center,
        rtol=0,
        atol=1e-10,
    )

    error = measured_center - expected_center

    print()
    print(f"Centering diagnostics: {name}")
    print("Mode:", AUTOCENTER_METHOD if AUTOCENTER else "manual")
    print("Applied detector shift (row, column):", DETECTOR_SHIFT_PX)
    print("Reference physical center:", reference_center)
    print("Expected shifted center:", expected_center)
    print("Estimated center in metadata:", measured_center)
    print("Center used by the geometry:", geometry_center)
    print("Estimated detector shift:", measured_center - reference_center)
    print("Center error vector:", error)
    print(f"Center error norm (px): {np.linalg.norm(error):.6f}")

    if not np.allclose(
        measured_center,
        expected_center,
        rtol=0,
        atol=1e-10,
    ):
        print(
            "WARNING: the center estimate differs from "
            "the known simulated center."
        )

    diagnostics = getattr(
        model.ptyscan,
        "centering_diagnostics",
        {},
    )

    for entry in diagnostics.values():
        if entry.get("boundary_hit", False):
            print(
                "WARNING: the estimate reached the search boundary."
            )

    return measured_center


def main() -> None:
    """Run the Siemens-star ER/HIO/RAAR comparison."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    shape = amplitude_from_image(IMAGE_PATH).shape

    if shape[0] != shape[1]:
        raise ValueError(
            "This tutorial requires a square input image."
        )

    if any(size % 2 for size in shape):
        raise ValueError(
            "This integer-center tutorial requires even frame dimensions."
        )

    true_object, known_support = make_complex_test_object(
        shape=shape,
        amplitude_model="image",
        amplitude_image_path=IMAGE_PATH,
        image_binary_threshold=0.5,
        support_radius_fraction=0.252,
    )

    geometry = CDIGeometry.from_parameters(
        shape=shape,
        energy_kev=ENERGY_KEV,
        distance_m=DISTANCE_M,
        detector_psize_m=DETECTOR_PSIZE_M,
        propagation="farfield",
        ffttype="numpy",
        data_type=np.complex128,
    )

    reference_intensity = np.abs(geometry.forward(true_object)) ** 2
    reference_center = np.asarray(shape, dtype=float) / 2.0

    shift = np.asarray(DETECTOR_SHIFT_PX, dtype=float)

    if (
        shift.shape != (2,)
        or not np.isfinite(shift).all()
        or not np.array_equal(shift, np.rint(shift))
    ):
        raise ValueError(
            "DETECTOR_SHIFT_PX must contain two finite integers."
        )

    expected_center = reference_center + shift

    if (
        np.any(expected_center < 0)
        or np.any(expected_center >= np.asarray(shape))
    ):
        raise ValueError(
            "Expected shifted center is outside the detector frame."
        )

    measured_intensity, detector_mask = shift_diffraction_pattern(
        reference_intensity,
        shift_px=DETECTOR_SHIFT_PX,
    )

    print("Simulated detector shift:", DETECTOR_SHIFT_PX, flush=True)
    print("Expected detector center:", expected_center, flush=True)
    print(
        "Centering is performed by each reconstruction's CDIScan.",
        flush=True,
    )

    # ER with the known support.
    er_engines = u.Param()
    er_engines.er = u.Param(
        name="CDIER",
        numiter=TOTAL_ITERATIONS,
        numiter_contiguous=1,
    )

    er_reconstruction = Ptycho(
        make_parameters(
            measured_intensity,
            known_support,
            er_engines,
            valid_mask=detector_mask,
        ),
        level=5,
    )

    # HIO with the same known support.
    hio_engines = u.Param()
    hio_engines.hio = u.Param(
        name="CDIHIO",
        numiter=TOTAL_ITERATIONS,
        numiter_contiguous=1,
        beta=HIO_BETA,
    )

    hio_reconstruction = Ptycho(
        make_parameters(
            measured_intensity,
            known_support,
            hio_engines,
            valid_mask=detector_mask,
        ),
        level=5,
    )

    # RAAR with the same known support.
    raar_engines = u.Param()
    raar_engines.raar = u.Param(
        name="CDIRAAR",
        numiter=TOTAL_ITERATIONS,
        numiter_contiguous=1,
        beta=RAAR_BETA,
    )

    raar_reconstruction = Ptycho(
        make_parameters(
            measured_intensity,
            known_support,
            raar_engines,
            valid_mask=detector_mask,
        ),
        level=5,
    )

    chosen_centers = [
        report_centering(
            name,
            reconstruction,
            reference_center,
            expected_center,
            measured_intensity,
            detector_mask,
        )
        for name, reconstruction in (
            ("ER", er_reconstruction),
            ("HIO", hio_reconstruction),
            ("RAAR", raar_reconstruction),
        )
    ]

    for center in chosen_centers[1:]:
        np.testing.assert_array_equal(center, chosen_centers[0])

    results = [
        result_from_ptycho("ER", er_reconstruction),
        result_from_ptycho(
            f"HIO, beta={HIO_BETA:g}",
            hio_reconstruction,
        ),
        result_from_ptycho(
            f"RAAR, beta={RAAR_BETA:g}",
            raar_reconstruction,
        ),
    ]

    save_reconstruction_figure(
        OUTPUT_DIRECTORY / "reconstruction_comparison.png",
        true_object=true_object,
        known_support=known_support,
        intensity=measured_intensity,
        results=results,
    )

    save_error_plot(
        OUTPUT_DIRECTORY / "error_curves.png",
        results,
    )

    print()
    print(f"Shape: {shape}")
    print(f"Iterations per method: {TOTAL_ITERATIONS}")
    print(f"Common support fraction: {known_support.mean():.4f}")

    for result in results:
        nrmse = complex_nrmse(
            true_object,
            result.object_field,
            known_support,
        )

        print()
        print(result.name)
        print(
            "  Last recorded amplitude residual: "
            f"{result.amplitude_errors[-1]:.4e}"
        )
        print(
            "  Complex NRMSE in known support "
            f"(global-phase alignment only): {nrmse:.4e}"
        )

    print()
    print(f"Figures saved to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()