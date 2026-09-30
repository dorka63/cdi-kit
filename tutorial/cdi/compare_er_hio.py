"""Compare ER, HIO, and staged HIO/ER on an oversampled Siemens-star CDI case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ptypy import io
from ptypy.custom.cdi_common import (
    CDIGeometry,
    CDIProblem,
    support_from_autocorrelation,
)
from ptypy.custom.cdi_er import ErrorReduction
from ptypy.custom.cdi_hio import HybridInputOutput
from ptypy.simulations.cdi_simulation import (
    amplitude_from_image,
    circular_support,
    make_complex_test_object,
)


Array = np.ndarray


@dataclass
class ReconstructionResult:
    """Store one reconstruction result and its iteration histories."""

    name: str
    object_field: Array
    amplitude_errors: list[float]
    object_changes: list[float]
    iterations: int
    support_name: str


def align_global_phase(
    reference: Array,
    reconstruction: Array,
    mask: Array,
) -> Array:
    """Align a reconstruction to the reference by its global phase."""
    overlap = np.vdot(
        reference[mask],
        reconstruction[mask],
    )

    if np.abs(overlap) == 0.0:
        return reconstruction.copy()

    return reconstruction * np.exp(1j * np.angle(overlap))


def complex_nrmse(
    reference: Array,
    reconstruction: Array,
    mask: Array,
) -> float:
    """Calculate complex NRMSE after alignment by a global phase."""
    aligned = align_global_phase(
        reference=reference,
        reconstruction=reconstruction,
        mask=mask,
    )

    numerator = np.linalg.norm(
        (aligned - reference)[mask]
    )
    denominator = np.linalg.norm(reference[mask])

    if denominator == 0.0:
        raise ValueError("Reference object is zero inside the evaluation mask.")

    return float(numerator / denominator)


def random_initial_guess(
    support: Array,
    seed: int,
) -> Array:
    """Return a random-phase field constrained to the supplied support."""
    rng = np.random.default_rng(seed)

    phase = rng.uniform(
        low=-np.pi,
        high=np.pi,
        size=support.shape,
    )

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
        amplitude_errors=list(solver.amplitude_errors),
        object_changes=list(solver.object_changes),
        iterations=iterations,
        support_name="known circular support",
    )


def run_hio(
    problem: CDIProblem,
    initial_object: Array,
    iterations: int,
    beta: float,
) -> ReconstructionResult:
    """Run fixed-beta Hybrid Input-Output with a fixed support."""
    solver = HybridInputOutput(
        problem=problem,
        object_field=initial_object.copy(),
        beta=beta,
    )
    solver.run(iterations)

    return ReconstructionResult(
        name=f"HIO, beta={beta:g}",
        object_field=solver.object_field.copy(),
        amplitude_errors=list(solver.amplitude_errors),
        object_changes=list(solver.object_changes),
        iterations=iterations,
        support_name="known circular support",
    )


def run_staged_hio_er(
    problem: CDIProblem,
    initial_object: Array,
    repetitions: int = 3,
) -> ReconstructionResult:
    """Run the requested 360-iteration staged HIO/ER schedule.

    Each repetition executes:

    - 30 HIO iterations with beta = 0.7, then 10 ER iterations;
    - 30 HIO iterations with beta = 0.4, then 10 ER iterations;
    - 30 HIO iterations with beta = 0.1, then 10 ER iterations.

    Ten repetitions yield 1,200 total iterations.
    """
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")

    schedule = (
	(30, 1.0, 10),
        (30, 0.7, 10),
        (30, 0.4, 10),
        (30, 0.1, 10),
    )

    object_field = initial_object.copy()
    amplitude_errors: list[float] = []
    object_changes: list[float] = []

    for _ in range(repetitions):
        for hio_iterations, beta, er_iterations in schedule:
            hio = HybridInputOutput(
                problem=problem,
                object_field=object_field,
                beta=beta,
            )
            hio.run(hio_iterations)

            object_field = hio.object_field.copy()
            amplitude_errors.extend(hio.amplitude_errors)
            object_changes.extend(hio.object_changes)

            er = ErrorReduction(
                problem=problem,
                object_field=object_field,
            )
            er.run(er_iterations)

            object_field = er.object_field.copy()
            amplitude_errors.extend(er.amplitude_errors)
            object_changes.extend(er.object_changes)

    return ReconstructionResult(
        name="Staged HIO/ER",
        object_field=object_field,
        amplitude_errors=amplitude_errors,
        object_changes=object_changes,
        iterations=len(amplitude_errors),
        support_name="autocorrelation support",
    )


def save_reconstruction_figure(
    output_path: Path,
    true_object: Array,
    known_support: Array,
    autocorrelation_support: Array,
    intensity: Array,
    results: list[ReconstructionResult],
) -> None:
    """Save reference data, both supports, and reconstruction diagnostics."""
    n_rows = 1 + len(results)

    figure, axes = plt.subplots(
        nrows=n_rows,
        ncols=5,
        figsize=(20, 4 * n_rows),
        constrained_layout=True,
    )

    reference_images = (
        (
            np.abs(true_object),
            "True amplitude",
            "viridis",
        ),
        (
            np.angle(true_object),
            "True phase",
            "twilight",
        ),
        (
            np.log10(intensity + 1e-12),
            "log10 far-field intensity",
            "magma",
        ),
        (
            known_support.astype(float),
            "Known circular support",
            "gray",
        ),
        (
            autocorrelation_support.astype(float),
            "Autocorrelation support",
            "gray",
        ),
    )

    for axis, (image, title, colormap) in zip(
        axes[0],
        reference_images,
        strict=True,
    ):
        artist = axis.imshow(image, cmap=colormap)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(artist, ax=axis, shrink=0.76)

    evaluation_mask = known_support

    for row, result in enumerate(results, start=1):
        aligned = align_global_phase(
            reference=true_object,
            reconstruction=result.object_field,
            mask=evaluation_mask,
        )

        images = (
            (
                np.abs(aligned),
                f"{result.name}: amplitude",
                "viridis",
            ),
            (
                np.angle(aligned),
                f"{result.name}: phase",
                "twilight",
            ),
            (
                np.abs(aligned - true_object),
                f"{result.name}: |complex error|",
                "magma",
            ),
            (
                np.abs(result.object_field),
                f"{result.name}: raw amplitude",
                "viridis",
            ),
            (
                result.object_field.real,
                f"{result.name}: real part",
                "RdBu_r",
            ),
        )

        for axis, (image, title, colormap) in zip(
            axes[row],
            images,
            strict=True,
        ):
            artist = axis.imshow(image, cmap=colormap)
            axis.set_title(title)
            axis.set_axis_off()
            figure.colorbar(artist, ax=axis, shrink=0.76)

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_error_plot(
    output_path: Path,
    results: list[ReconstructionResult],
) -> None:
    """Save detector residual and object-update histories."""
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 4.8),
        constrained_layout=True,
    )

    for result in results:
        iterations = np.arange(
            1,
            len(result.amplitude_errors) + 1,
        )

        label = (
            f"{result.name} "
            f"({result.support_name})"
        )

        axes[0].semilogy(
            iterations,
            result.amplitude_errors,
            label=label,
        )

        axes[1].semilogy(
            iterations,
            result.object_changes,
            label=label,
        )

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Masked amplitude residual")
    axes[0].set_title("Data-constraint error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Relative object change")
    axes[1].set_title("Update magnitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_arrays(
    output_directory: Path,
    true_object: Array,
    known_support: Array,
    autocorrelation_support: Array,
    intensity: Array,
    results: list[ReconstructionResult],
) -> None:
    """Save simulation inputs, supports, reconstructions, and histories."""
    np.save(output_directory / "true_object.npy", true_object)
    np.save(output_directory / "known_support.npy", known_support)
    np.save(
        output_directory / "autocorrelation_support.npy",
        autocorrelation_support,
    )
    np.save(output_directory / "measured_intensity.npy", intensity)

    filenames = (
        "er",
        "hio_beta_0p5",
        "staged_hio_er",
    )

    for result, filename in zip(results, filenames, strict=True):
        np.save(
            output_directory / f"{filename}_reconstruction.npy",
            result.object_field,
        )
        np.save(
            output_directory / f"{filename}_amplitude_errors.npy",
            np.asarray(result.amplitude_errors),
        )
        np.save(
            output_directory / f"{filename}_object_changes.npy",
            np.asarray(result.object_changes),
        )


def main() -> None:
    """Run the Siemens-star ER/HIO comparison."""
    image_path = Path("tutorial/cdi/data/star.png")
    baseline_iterations = 320
    fixed_hio_beta = 0.5
    staged_repetitions = 2
    seed = 2026

    output_directory = Path("output") / "cdi_er_hio_example"
    output_directory.mkdir(parents=True, exist_ok=True)

    image, _metadata = io.image_read(str(image_path))
    shape = tuple(image.shape[:2])

    amplitude = amplitude_from_image(image_path)

    if amplitude.shape != shape:
        raise RuntimeError(
            "The amplitude-map shape differs from the image shape."
        )

    known_support = circular_support(
        shape=shape,
        radius_px=min(shape) / 4.0,
    )

    true_object, simulation_support = make_complex_test_object(
        shape=shape,
        amplitude_model="image",
        amplitude_image_path=image_path,
        image_invert=False,
        amplitude_floor=0.0,
        amplitude_ceiling=1.0,
        support_radius_fraction=0.25,
    )

    if not np.array_equal(known_support, simulation_support):
        raise RuntimeError(
            "Known support and simulation support must be identical."
        )

    geometry = CDIGeometry.from_parameters(
        shape=shape,
        energy_kev=8.0,
        distance_m=1.0,
        detector_psize_m=55e-6,
        propagation="farfield",
        ffttype="numpy",
        data_type=np.complex128,
    )

    measured_intensity = np.abs(
        geometry.forward(true_object)
    ) ** 2

    autocorrelation_support = support_from_autocorrelation(
        measured_intensity,
        gaussian_sigma_px=5.0,
        threshold=0.30,
        closing_iterations=2,
    )

    known_support_problem = CDIProblem(
        geometry=geometry,
        measured_intensity=measured_intensity,
        valid_mask=np.ones(shape, dtype=bool),
        support=known_support,
    )

    autocorrelation_problem = CDIProblem(
        geometry=geometry,
        measured_intensity=measured_intensity,
        valid_mask=np.ones(shape, dtype=bool),
        support=autocorrelation_support,
    )

    known_initial_guess = random_initial_guess(
        support=known_support,
        seed=seed,
    )

    autocorrelation_initial_guess = random_initial_guess(
        support=autocorrelation_support,
        seed=seed,
    )

    initial_known_error = known_support_problem.data_projection(
        known_initial_guess
    )[1]

    initial_autocorrelation_error = autocorrelation_problem.data_projection(
        autocorrelation_initial_guess
    )[1]

    er_result = run_er(
        problem=known_support_problem,
        initial_object=known_initial_guess,
        iterations=baseline_iterations,
    )

    hio_result = run_hio(
        problem=known_support_problem,
        initial_object=known_initial_guess,
        iterations=baseline_iterations,
        beta=fixed_hio_beta,
    )

    staged_result = run_staged_hio_er(
        problem=autocorrelation_problem,
        initial_object=autocorrelation_initial_guess,
        repetitions=staged_repetitions,
    )

    results = [
        er_result,
        hio_result,
        staged_result,
    ]

    save_reconstruction_figure(
        output_path=output_directory / "reconstruction_comparison.png",
        true_object=true_object,
        known_support=known_support,
        autocorrelation_support=autocorrelation_support,
        intensity=measured_intensity,
        results=results,
    )

    save_error_plot(
        output_path=output_directory / "error_curves.png",
        results=results,
    )

    save_arrays(
        output_directory=output_directory,
        true_object=true_object,
        known_support=known_support,
        autocorrelation_support=autocorrelation_support,
        intensity=measured_intensity,
        results=results,
    )

    print("Synthetic Siemens-star CDI comparison complete")
    print(f"Image path: {image_path}")
    print(f"Simulation shape: {shape}")
    print(f"Known-support fraction: {known_support.mean():.6f}")
    print(
        "Autocorrelation-support fraction: "
        f"{autocorrelation_support.mean():.6f}"
    )
    print(
        "Initial amplitude residual, known support: "
        f"{initial_known_error:.6e}"
    )
    print(
        "Initial amplitude residual, autocorrelation support: "
        f"{initial_autocorrelation_error:.6e}"
    )

    for result in results:
        final_problem = (
            known_support_problem
            if result.support_name == "known circular support"
            else autocorrelation_problem
        )

        final_error = final_problem.data_projection(
            result.object_field
        )[1]

        nrmse = complex_nrmse(
            reference=true_object,
            reconstruction=result.object_field,
            mask=known_support,
        )

        print()
        print(result.name)
        print(f"  Support: {result.support_name}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Final amplitude residual: {final_error:.6e}")
        print(f"  Complex NRMSE in known support: {nrmse:.6e}")

    print()
    print(f"Output directory: {output_directory}")


if __name__ == "__main__":
    main()