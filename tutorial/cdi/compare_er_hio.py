"""Compare CPU Error Reduction and HIO on a synthetic far-field CDI problem."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ptypy.custom.cdi_common import CDIGeometry, CDIProblem
from ptypy.custom.cdi_er import ErrorReduction
from ptypy.custom.cdi_hio import HybridInputOutput
from ptypy.simulations.cdi_simulation import make_complex_test_object


def align_global_phase(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Align reconstruction to reference by an unobservable global phase."""
    overlap = np.vdot(
        reference[support],
        reconstruction[support],
    )

    if np.abs(overlap) == 0.0:
        return reconstruction.copy()

    phase = np.angle(overlap)

    return reconstruction * np.exp(1j * phase)


def complex_nrmse(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    support: np.ndarray,
) -> float:
    """Compute object error after global-phase alignment."""
    aligned = align_global_phase(reference, reconstruction, support)

    numerator = np.linalg.norm(
        (aligned - reference)[support]
    )
    denominator = np.linalg.norm(reference[support])

    return float(numerator / denominator)


def random_initial_guess(
    support: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Create a unit-amplitude random-phase starting field inside support."""
    rng = np.random.default_rng(seed)

    phase = rng.uniform(
        low=-np.pi,
        high=np.pi,
        size=support.shape,
    )

    return support.astype(np.complex128) * np.exp(1j * phase)


def save_figure(
    output_path: Path,
    true_object: np.ndarray,
    support: np.ndarray,
    intensity: np.ndarray,
    er: ErrorReduction,
    hio: HybridInputOutput,
) -> None:
    """Create a compact diagnostic comparison figure."""
    er_aligned = align_global_phase(
        true_object,
        er.object_field,
        support,
    )
    hio_aligned = align_global_phase(
        true_object,
        hio.object_field,
        support,
    )

    figure, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(16, 12),
        constrained_layout=True,
    )

    image_specs = [
        (
            axes[0, 0],
            np.abs(true_object),
            "True amplitude",
            "viridis",
        ),
        (
            axes[0, 1],
            np.angle(true_object),
            "True phase",
            "twilight",
        ),
        (
            axes[0, 2],
            np.log10(intensity + 1e-12),
            "log10 far-field intensity",
            "magma",
        ),
        (
            axes[0, 3],
            support.astype(float),
            "Known support",
            "gray",
        ),
        (
            axes[1, 0],
            np.abs(er_aligned),
            "ER amplitude",
            "viridis",
        ),
        (
            axes[1, 1],
            np.angle(er_aligned),
            "ER phase",
            "twilight",
        ),
        (
            axes[1, 2],
            np.abs(er_aligned - true_object),
            "ER amplitude of complex error",
            "magma",
        ),
        (
            axes[1, 3],
            np.abs(er.object_field),
            "ER raw amplitude",
            "viridis",
        ),
        (
            axes[2, 0],
            np.abs(hio_aligned),
            "HIO amplitude",
            "viridis",
        ),
        (
            axes[2, 1],
            np.angle(hio_aligned),
            "HIO phase",
            "twilight",
        ),
        (
            axes[2, 2],
            np.abs(hio_aligned - true_object),
            "HIO amplitude of complex error",
            "magma",
        ),
        (
            axes[2, 3],
            np.abs(hio.object_field),
            "HIO raw amplitude",
            "viridis",
        ),
    ]

    for axis, image, title, colormap in image_specs:
        artist = axis.imshow(image, cmap=colormap)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(artist, ax=axis, shrink=0.78)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_error_plot(
    output_path: Path,
    er: ErrorReduction,
    hio: HybridInputOutput,
) -> None:
    """Save detector-amplitude-error and object-change trajectories."""
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4.5),
        constrained_layout=True,
    )

    axes[0].semilogy(
        er.amplitude_errors,
        label="ER",
    )
    axes[0].semilogy(
        hio.amplitude_errors,
        label="HIO",
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Masked amplitude residual")
    axes[0].set_title("Data-constraint error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(
        er.object_changes,
        label="ER",
    )
    axes[1].semilogy(
        hio.object_changes,
        label="HIO",
    )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Relative object change")
    axes[1].set_title("Update magnitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    shape = (128, 128)
    n_iterations = 500
    beta = 0.9
    seed = 2026

    output_directory = Path("output") / "cdi_er_hio_example"
    output_directory.mkdir(parents=True, exist_ok=True)

    true_object, support = make_complex_test_object(shape)

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

    problem = CDIProblem(
        geometry=geometry,
        measured_intensity=measured_intensity,
        valid_mask=np.ones(shape, dtype=bool),
        support=support,
    )

    initial_guess = random_initial_guess(
        support=support,
        seed=seed,
    )

    er = ErrorReduction(
        problem=problem,
        object_field=initial_guess.copy(),
    )

    hio = HybridInputOutput(
        problem=problem,
        object_field=initial_guess.copy(),
        beta=beta,
    )

    initial_error = problem.data_projection(initial_guess)[1]

    er.run(n_iterations)
    hio.run(n_iterations)

    er_final_error = problem.data_projection(er.object_field)[1]
    hio_final_error = problem.data_projection(hio.object_field)[1]

    er_nrmse = complex_nrmse(
        true_object,
        er.object_field,
        support,
    )

    hio_nrmse = complex_nrmse(
        true_object,
        hio.object_field,
        support,
    )

    save_figure(
        output_directory / "reconstruction_comparison.png",
        true_object=true_object,
        support=support,
        intensity=measured_intensity,
        er=er,
        hio=hio,
    )

    save_error_plot(
        output_directory / "error_curves.png",
        er=er,
        hio=hio,
    )

    np.save(
        output_directory / "true_object.npy",
        true_object,
    )
    np.save(
        output_directory / "support.npy",
        support,
    )
    np.save(
        output_directory / "measured_intensity.npy",
        measured_intensity,
    )
    np.save(
        output_directory / "er_reconstruction.npy",
        er.object_field,
    )
    np.save(
        output_directory / "hio_reconstruction.npy",
        hio.object_field,
    )

    print("Synthetic CDI ER/HIO comparison complete")
    print(f"Iterations: {n_iterations}")
    print(f"HIO beta: {beta}")
    print(f"Initial amplitude residual: {initial_error:.6e}")
    print(f"ER final amplitude residual: {er_final_error:.6e}")
    print(f"HIO final amplitude residual: {hio_final_error:.6e}")
    print(f"ER complex NRMSE inside support: {er_nrmse:.6e}")
    print(f"HIO complex NRMSE inside support: {hio_nrmse:.6e}")
    print(f"Output directory: {output_directory}")


if __name__ == "__main__":
    main()