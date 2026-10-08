"""Compare PtyPy CDI ER, HIO, and HIO with shrink-wrap on a Siemens star."""

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
    support_from_autocorrelation,
)
import ptypy.custom.cdi_er
import ptypy.custom.cdi_hio
import ptypy.custom.cdi_shrinkwrap
from ptypy.simulations.cdi_simulation import (
    amplitude_from_image,
    make_complex_test_object,
    shift_diffraction_pattern,
)


Array = np.ndarray

IMAGE_PATH = Path("tutorial/cdi/data/star.png")
OUTPUT_DIRECTORY = Path("output") / "cdi_er_hio_example"

ENERGY_KEV = 8.0
DISTANCE_M = 1.0
DETECTOR_PSIZE_M = 55e-6

TOTAL_ITERATIONS = 360
HIO_BETA = 0.5

DETECTOR_SHIFT_PX = (7, -5)
AUTOCENTER = True
AUTOCENTER_METHOD = "reflections"
AUTOCENTER_INITIAL_CENTER = None
AUTOCENTER_SEARCH_RADIUS_PX = 12.0
AUTOCENTER_TOLERANCE = 1e-4
AUTOCENTER_MIN_OVERLAP = 0.5
MANUAL_CENTER_PX = None

AUTOCORRELATION_THRESHOLD = 0.15
AUTOCORRELATION_BLUR_SIGMA_PX = None
AUTOCORRELATION_CLOSING_ITERATIONS = 0

SHRINKWRAP_BETA = 0.7
SHRINKWRAP_UPDATE_INTERVAL = 20
SHRINKWRAP_FIRST_UPDATE = 5
SHRINKWRAP_SIGMA_START_PX = 3.0
SHRINKWRAP_SIGMA_DECAY = 0.99
SHRINKWRAP_SIGMA_MIN_PX = 1.5
SHRINKWRAP_THRESHOLD = 0.3
SHRINKWRAP_CLOSING_ITERATIONS = 1
SHRINKWRAP_FINAL_ER_ITERATIONS = 100


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
    """Remove the unobservable global phase from a reconstruction."""
    overlap = np.vdot(reference[mask], reconstruction[mask])

    if np.abs(overlap) == 0.0:
        return reconstruction.copy()

    return reconstruction * np.exp(1j * np.angle(overlap))


def complex_nrmse(
    reference: Array,
    reconstruction: Array,
    mask: Array,
) -> float:
    """Return complex NRMSE inside ``mask`` after global-phase alignment."""
    aligned = align_global_phase(reference, reconstruction, mask)
    numerator = np.linalg.norm((aligned - reference)[mask])
    denominator = np.linalg.norm(reference[mask])
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
    true_phase: Array,
    known_support: Array,
    autocorrelation_support: Array,
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

    show(axes[0, 0], figure, np.abs(true_object), "True amplitude", "viridis")
    show(
        axes[0, 1],
        figure,
        np.ma.masked_where(~known_support, true_phase),
        "True phase",
        "twilight",
    )
    show(
        axes[0, 2],
        figure,
        np.log10(intensity + 1e-12),
        "log10 far-field intensity",
        "magma",
    )
    show(
        axes[0, 3],
        figure,
        autocorrelation_support.astype(float),
        "Initial autocorrelation support",
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
    """Save Fourier residual and relative object-change histories."""
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
            result.amplitude_errors,
            label=result.name,
        )
        axes[1].semilogy(
            iterations,
            result.object_changes,
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
    """Create a PtyPy configuration for one single-frame CDI reconstruction."""
    if not AUTOCENTER and MANUAL_CENTER_PX is None:
        raise ValueError("Set MANUAL_CENTER_PX when AUTOCENTER is False.")

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
    parameters.scans.cdi = u.Param(name="Vanilla")
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
    """Extract the final CDI object, support, and histories from PtyPy."""
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
    """Report the center chosen by the reconstruction's own CDIScan."""
    model = reconstruction.model.scans["cdi"]
    measured_center = np.asarray(model.ptyscan.meta.center, dtype=float)
    geometry_center = np.asarray(model.geometries[0].p.center, dtype=float)
    pod = next(iter(reconstruction.pods.values()))
    np.testing.assert_array_equal(pod.diff, input_intensity)
    np.testing.assert_array_equal(pod.mask, input_mask)
    np.testing.assert_allclose(geometry_center, measured_center, rtol=0, atol=1e-10)
    error = measured_center-expected_center
    print()
    print(f"Centering diagnostics: {name}")
    print("Mode:", AUTOCENTER_METHOD if AUTOCENTER else "manual")
    print("Applied detector shift (row, column):", DETECTOR_SHIFT_PX)
    print("Reference physical center:", reference_center)
    print("Expected shifted center:", expected_center)
    print("Estimated center in metadata:", measured_center)
    print("Center used by the geometry:", geometry_center)
    print("Estimated detector shift:", measured_center-reference_center)
    print("Center error vector:", error)
    print(f"Center error norm (px): {np.linalg.norm(error):.6f}")
    if not np.allclose(measured_center, expected_center, rtol=0, atol=1e-10):
        print("WARNING: the center estimate differs from the known simulated center.")
    for entry in getattr(model.ptyscan, "centering_diagnostics", {}).values():
        if entry.get("boundary_hit", False):
            print("WARNING: the estimate reached the search boundary.")
    return measured_center


def main() -> None:
    """Run the Siemens-star comparison."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # The simulation uses the same PtyPy propagator as reconstruction.
    shape = amplitude_from_image(IMAGE_PATH).shape

    true_object, known_support = make_complex_test_object(
        shape=shape,
        amplitude_model="image",
        amplitude_image_path=IMAGE_PATH,
        image_binary_threshold=0.5,
        support_radius_fraction=0.25,
    )

    phase_reference, _ = make_complex_test_object(
        shape=shape,
        amplitude_model="uniform",
        support_radius_fraction=0.25,
    )
    true_phase = np.angle(phase_reference)

    geometry = CDIGeometry.from_parameters(
        shape=shape,
        energy_kev=ENERGY_KEV,
        distance_m=DISTANCE_M,
        detector_psize_m=DETECTOR_PSIZE_M,
        propagation="farfield",
        ffttype="numpy",
        data_type=np.complex128,
    )
    if any(size % 2 for size in shape):
        raise ValueError("This integer-center tutorial requires even frame dimensions.")
    reference_intensity = np.abs(geometry.forward(true_object)) ** 2
    reference_center = np.asarray(shape, dtype=float) / 2.0
    shift = np.asarray(DETECTOR_SHIFT_PX, dtype=float)
    if shift.shape != (2,) or not np.isfinite(shift).all() or not np.array_equal(shift, np.rint(shift)):
        raise ValueError("DETECTOR_SHIFT_PX must contain two finite integers.")
    expected_center = reference_center + shift
    if np.any(expected_center < 0) or np.any(expected_center >= np.asarray(shape)):
        raise ValueError("Expected shifted center is outside the detector frame.")
    measured_intensity, detector_mask = shift_diffraction_pattern(
        reference_intensity,
        shift_px=DETECTOR_SHIFT_PX,
    )
    print("Simulated detector shift:", DETECTOR_SHIFT_PX, flush=True)
    print("Expected detector center:", expected_center, flush=True)
    print("Centering is performed by each reconstruction's CDIScan.", flush=True)

    # A cyclic integer shift multiplies the inverse FFT by a phase ramp.
    # Its magnitude, used by support_from_autocorrelation, is unchanged.
    reference_ac = np.abs(np.fft.ifft2(np.fft.ifftshift(reference_intensity)))
    shifted_ac = np.abs(np.fft.ifft2(np.fft.ifftshift(measured_intensity)))
    if not np.allclose(reference_ac, shifted_ac, rtol=1e-9,
                       atol=1e-12 * max(1.0, float(reference_ac.max()))):
        raise RuntimeError("Autocorrelation magnitude changed under the cyclic shift.")

    autocorrelation_support = support_from_autocorrelation(
        measured_intensity,
        valid_mask=detector_mask,
        gaussian_sigma_px=AUTOCORRELATION_BLUR_SIGMA_PX,
        threshold=AUTOCORRELATION_THRESHOLD,
        closing_iterations=AUTOCORRELATION_CLOSING_ITERATIONS,
    )

    # ER with the known circular support.
    er_engines = u.Param()
    er_engines.er = u.Param(
        name="CDIER",
        numiter=TOTAL_ITERATIONS,
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

    # HIO + shrink-wrap starts only with autocorrelation support.
    #
    # PtyPy executes engines in lexicographic order of their labels.
    # A shared zero-padded counter therefore defines the algorithm order:
    #
    # HIO block -> shrink-wrap -> HIO block -> shrink-wrap -> ... -> ER.
    shrinkwrap_engines = u.Param()
    hio_iterations = TOTAL_ITERATIONS - SHRINKWRAP_FINAL_ER_ITERATIONS
    completed_hio_iterations = 0
    shrinkwrap_updates = 0
    sigma = SHRINKWRAP_SIGMA_START_PX
    engine_index = 0

    while completed_hio_iterations < hio_iterations:
        hio_block = min(
            SHRINKWRAP_UPDATE_INTERVAL,
            hio_iterations - completed_hio_iterations,
        )

        shrinkwrap_engines[f"e{engine_index:03d}_hio"] = u.Param(
            name="CDIHIO",
            numiter=hio_block,
            beta=SHRINKWRAP_BETA,
        )
        engine_index += 1
        completed_hio_iterations += hio_block

        if (
            SHRINKWRAP_FIRST_UPDATE <= completed_hio_iterations
            and completed_hio_iterations < hio_iterations
        ):
            shrinkwrap_engines[
                f"e{engine_index:03d}_shrinkwrap"
            ] = u.Param(
                name="CDIShrinkWrap",
                numiter=1,
                gaussian_sigma_px=sigma,
                threshold=SHRINKWRAP_THRESHOLD,
                sigma_decay=1.0,
                minimum_sigma_px=SHRINKWRAP_SIGMA_MIN_PX,
                closing_iterations=SHRINKWRAP_CLOSING_ITERATIONS,
            )
            engine_index += 1

            sigma = max(
                SHRINKWRAP_SIGMA_MIN_PX,
                sigma * SHRINKWRAP_SIGMA_DECAY,
            )
            shrinkwrap_updates += 1

    shrinkwrap_engines[f"e{engine_index:03d}_er_final"] = u.Param(
        name="CDIER",
        numiter=SHRINKWRAP_FINAL_ER_ITERATIONS,
    )
    shrinkwrap_reconstruction = Ptycho(
        make_parameters(
            measured_intensity,
            autocorrelation_support,
            shrinkwrap_engines,
            valid_mask=detector_mask,
        ),
        level=5,
    )

    chosen_centers = [
        report_centering(
            name, reconstruction, reference_center, expected_center,
            measured_intensity, detector_mask,
        )
        for name, reconstruction in (
            ("ER", er_reconstruction),
            ("HIO", hio_reconstruction),
            ("HIO + shrink-wrap", shrinkwrap_reconstruction),
        )
    ]
    for center in chosen_centers[1:]:
        np.testing.assert_array_equal(center, chosen_centers[0])

    results = [
        result_from_ptycho("ER", er_reconstruction),
        result_from_ptycho(f"HIO, beta={HIO_BETA:g}", hio_reconstruction),
        result_from_ptycho("HIO + shrink-wrap", shrinkwrap_reconstruction),
    ]

    save_reconstruction_figure(
        OUTPUT_DIRECTORY / "reconstruction_comparison.png",
        true_object=true_object,
        true_phase=true_phase,
        known_support=known_support,
        autocorrelation_support=autocorrelation_support,
        intensity=measured_intensity,
        results=results,
    )
    save_error_plot(OUTPUT_DIRECTORY / "error_curves.png", results)

    print()
    print(f"Shape: {shape}, iterations per method: {TOTAL_ITERATIONS}")
    print(
        "Support fraction: "
        f"known {known_support.mean():.4f}, "
        f"autocorrelation {autocorrelation_support.mean():.4f}, "
        f"final shrink-wrap {results[2].support.mean():.4f}"
    )
    print(
        "Shrink-wrap support updates: "
        f"{shrinkwrap_updates}; "
        f"next sigma would be {sigma:.4f} px."
    )

    for result in results:
        nrmse = complex_nrmse(
            true_object,
            result.object_field,
            known_support,
        )
        print()
        print(result.name)
        print(
            f"  Final amplitude residual: "
            f"{result.amplitude_errors[-1]:.4e}"
        )
        print(f"  Complex NRMSE in known support: {nrmse:.4e}")

    print()
    print(f"Figures saved to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()