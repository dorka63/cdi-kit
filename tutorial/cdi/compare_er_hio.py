"""Compare ER, HIO, and HIO with shrink-wrap on a synthetic Siemens star."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ptypy.custom.cdi_common import (
    CDIGeometry,
    CDIProblem,
    support_from_autocorrelation,
)
from ptypy.custom.cdi_er import ErrorReduction
from ptypy.custom.cdi_hio import HybridInputOutput
from ptypy.custom.cdi_shrinkwrap import ShrinkWrapSupport
from ptypy.simulations.cdi_simulation import (
    amplitude_from_image,
    make_complex_test_object,
)


Array = np.ndarray

IMAGE_PATH = Path("tutorial/cdi/data/star.png")
OUTPUT_DIRECTORY = Path("output") / "cdi_er_hio_example"

TOTAL_ITERATIONS = 360
HIO_BETA = 0.5
SEED = 2026

AUTOCORRELATION_THRESHOLD = 0.15
AUTOCORRELATION_BLUR_SIGMA_PX = None
AUTOCORRELATION_CLOSING_ITERATIONS = 0

SHRINKWRAP_BETA = 0.5
SHRINKWRAP_UPDATE_INTERVAL = 5
SHRINKWRAP_FIRST_UPDATE = 5
SHRINKWRAP_SIGMA_START_PX = 2.0
SHRINKWRAP_SIGMA_DECAY = 0.99
SHRINKWRAP_SIGMA_MIN_PX = 1.0
SHRINKWRAP_THRESHOLD = 0.1
SHRINKWRAP_CLOSING_ITERATIONS = 1
SHRINKWRAP_FINAL_ER_ITERATIONS = 60



@dataclass
class ReconstructionResult:
    """Reconstructed field, its final support, and iteration histories."""

    name: str
    object_field: Array
    support: Array
    amplitude_errors: list[float]
    object_changes: list[float]


def random_initial_guess(support: Array, seed: int) -> Array:
    """Return a unit-amplitude random-phase field inside support."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, size=support.shape)
    return support.astype(np.complex128) * np.exp(1j * phase)


def run_er(
    problem: CDIProblem,
    initial_object: Array,
    iterations: int,
) -> ReconstructionResult:
    """Run Error Reduction with a fixed support."""
    solver = ErrorReduction(
        problem=problem,
        object_field=initial_object.copy(),
    )
    solver.run(iterations)

    return ReconstructionResult(
        name="ER",
        object_field=solver.object_field.copy(),
        support=problem.support.copy(),
        amplitude_errors=list(solver.amplitude_errors),
        object_changes=list(solver.object_changes),
    )


def run_hio(
    problem: CDIProblem,
    initial_object: Array,
    iterations: int,
    beta: float,
) -> ReconstructionResult:
    """Run HIO with a fixed support."""
    solver = HybridInputOutput(
        problem=problem,
        object_field=initial_object.copy(),
        beta=beta,
    )
    solver.run(iterations)

    return ReconstructionResult(
        name=f"HIO, beta={beta:g}",
        object_field=solver.object_field.copy(),
        support=problem.support.copy(),
        amplitude_errors=list(solver.amplitude_errors),
        object_changes=list(solver.object_changes),
    )


def run_hio_shrinkwrap(
    problem: CDIProblem,
    initial_object: Array,
    total_iterations: int,
) -> ReconstructionResult:
    """Run HIO with periodic shrink-wrap updates, then final ER.

    The initial ``problem.support`` is the autocorrelation support and is
    replaced in place during the run.
    """
    hio_iterations = total_iterations - SHRINKWRAP_FINAL_ER_ITERATIONS

    shrinkwrap = ShrinkWrapSupport(
        gaussian_sigma_px=SHRINKWRAP_SIGMA_START_PX,
        threshold=SHRINKWRAP_THRESHOLD,
        sigma_decay=SHRINKWRAP_SIGMA_DECAY,
        minimum_sigma_px=SHRINKWRAP_SIGMA_MIN_PX,
        closing_iterations=SHRINKWRAP_CLOSING_ITERATIONS,
    )

    hio = HybridInputOutput(
        problem=problem,
        object_field=initial_object.copy(),
        beta=SHRINKWRAP_BETA,
    )

    completed = 0

    while completed < hio_iterations:
        block = min(SHRINKWRAP_UPDATE_INTERVAL, hio_iterations - completed)
        hio.run(block)
        completed += block

        if SHRINKWRAP_FIRST_UPDATE <= completed < hio_iterations:
            problem.set_support(shrinkwrap.update(hio.object_field))

    amplitude_errors = list(hio.amplitude_errors)
    object_changes = list(hio.object_changes)
    object_field = hio.object_field.copy()

    if SHRINKWRAP_FINAL_ER_ITERATIONS > 0:
        er = ErrorReduction(problem=problem, object_field=object_field)
        er.run(SHRINKWRAP_FINAL_ER_ITERATIONS)
        object_field = er.object_field.copy()
        amplitude_errors.extend(er.amplitude_errors)
        object_changes.extend(er.object_changes)

    print(
        f"Shrink-wrap: {shrinkwrap.update_count} support updates, "
        f"final sigma {shrinkwrap.current_sigma_px} px"
    )

    return ReconstructionResult(
        name="HIO + shrink-wrap",
        object_field=object_field,
        support=problem.support.copy(),
        amplitude_errors=amplitude_errors,
        object_changes=object_changes,
    )


def align_global_phase(
    reference: Array,
    reconstruction: Array,
    mask: Array,
) -> Array:
    """Remove the unobservable global phase."""
    overlap = np.vdot(reference[mask], reconstruction[mask])

    if np.abs(overlap) == 0.0:
        return reconstruction.copy()

    return reconstruction * np.exp(1j * np.angle(overlap))


def complex_nrmse(
    reference: Array,
    reconstruction: Array,
    mask: Array,
) -> float:
    """Complex NRMSE inside mask after global-phase alignment."""
    aligned = align_global_phase(reference, reconstruction, mask)
    numerator = np.linalg.norm((aligned - reference)[mask])
    denominator = np.linalg.norm(reference[mask])
    return float(numerator / denominator)


def show(axis, figure, image, title, colormap) -> None:
    """Draw one image panel with a title and a colour bar."""
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
    """Save the reference data and all reconstructions."""
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

        show(axes[row, 0], figure, np.abs(aligned),
             f"{result.name}: amplitude", "viridis")
        show(
            axes[row, 1],
            figure,
            np.ma.masked_where(~result.support, np.angle(aligned)),
            f"{result.name}: phase",
            "twilight",
        )
        show(axes[row, 2], figure, np.abs(aligned - true_object),
             f"{result.name}: |complex error|", "magma")
        show(axes[row, 3], figure, result.support.astype(float),
             f"{result.name}: support", "gray")

    figure.savefig(str(output_path.resolve()), dpi=150)
    plt.close(figure)


def save_error_plot(
    output_path: Path,
    results: list[ReconstructionResult],
) -> None:
    """Save amplitude-residual and object-change histories."""
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 4.8),
        constrained_layout=True,
    )

    for result in results:
        iterations = np.arange(1, len(result.amplitude_errors) + 1)
        axes[0].semilogy(iterations, result.amplitude_errors, label=result.name)
        axes[1].semilogy(iterations, result.object_changes, label=result.name)

    axes[0].set_title("Masked amplitude residual")
    axes[1].set_title("Relative object change")

    for axis in axes:
        axis.set_xlabel("Iteration")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.savefig(str(output_path.resolve()), dpi=150)
    plt.close(figure)


def main() -> None:
    """Run the Siemens-star comparison."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

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
        energy_kev=8.0,
        distance_m=1.0,
        detector_psize_m=55e-6,
        propagation="farfield",
        ffttype="numpy",
        data_type=np.complex128,
    )

    measured_intensity = np.abs(geometry.forward(true_object)) ** 2
    valid_mask = np.ones(shape, dtype=bool)

    autocorrelation_support = support_from_autocorrelation(
        measured_intensity,
        gaussian_sigma_px=AUTOCORRELATION_BLUR_SIGMA_PX,
        threshold=AUTOCORRELATION_THRESHOLD,
        closing_iterations=AUTOCORRELATION_CLOSING_ITERATIONS,
    )

    known_problem = CDIProblem(
        geometry=geometry,
        measured_intensity=measured_intensity,
        valid_mask=valid_mask,
        support=known_support,
    )

    shrinkwrap_problem = CDIProblem(
        geometry=geometry,
        measured_intensity=measured_intensity,
        valid_mask=valid_mask,
        support=autocorrelation_support,
    )

    known_guess = random_initial_guess(known_support, SEED)
    shrinkwrap_guess = random_initial_guess(autocorrelation_support, SEED)

    results = [
        run_er(known_problem, known_guess, TOTAL_ITERATIONS),
        run_hio(known_problem, known_guess, TOTAL_ITERATIONS, HIO_BETA),
        run_hio_shrinkwrap(shrinkwrap_problem, shrinkwrap_guess, TOTAL_ITERATIONS),
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

    for result in results:
        nrmse = complex_nrmse(true_object, result.object_field, known_support)
        print()
        print(result.name)
        print(f"  Final amplitude residual: {result.amplitude_errors[-1]:.4e}")
        print(f"  Complex NRMSE in known support: {nrmse:.4e}")

    print()
    print(f"Figures saved to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()