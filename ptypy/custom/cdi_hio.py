"""CPU Hybrid Input-Output reconstruction for coherent diffraction imaging."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cdi_common import register_citation
from .cdi_common import CDIProblem


Array = np.ndarray

HIO_ARTICLE = dict(
    comment="The Hybrid Input-Output phase-retrieval algorithm",
    title="Phase retrieval algorithms: a comparison",
    author="Fienup J. R.",
    journal="Applied Optics",
    volume=21,
    year=1982,
    page=2758,
    doi="10.1364/AO.21.002758",
)


@dataclass
class HybridInputOutput:
    """Classical support-only Hybrid Input-Output CDI reconstruction.

    The update rule is

    .. math::

        \\rho_{k+1}(r) =
        \\begin{cases}
        g_k(r), & r \\in S, \\\\
        \\rho_k(r) - \\beta g_k(r), & r \\notin S,
        \\end{cases}

    where ``g_k = P_M(rho_k)``.
    """

    problem: CDIProblem
    object_field: Array
    beta: float = 0.9
    amplitude_errors: list[float] = field(default_factory=list)
    object_changes: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must be in the interval (0, 1].")

        object_field = np.asarray(self.object_field)

        if object_field.shape != self.problem.geometry.shape:
            raise ValueError(
                "object_field shape must match geometry shape; "
                f"got {object_field.shape} and "
                f"{self.problem.geometry.shape}."
            )

        if not np.iscomplexobj(object_field):
            object_field = object_field.astype(np.complex128)

        self.object_field = object_field.astype(np.complex128, copy=True)
        register_citation(HIO_ARTICLE)

    def step(self) -> float:
        """Perform one HIO iteration and return its data error."""
        current = self.object_field

        candidate, amplitude_error = self.problem.data_projection(current)

        support = self.problem.support

        updated = current.copy()
        updated[support] = candidate[support]
        updated[~support] = (
            current[~support] - self.beta * candidate[~support]
        )

        denominator = np.linalg.norm(current)
        change = np.linalg.norm(updated - current)
        relative_change = (
            float(change / denominator)
            if denominator > 0.0
            else float(change)
        )

        self.object_field = updated
        self.amplitude_errors.append(amplitude_error)
        self.object_changes.append(relative_change)

        return amplitude_error

    def run(self, n_iterations: int) -> Array:
        """Run ``n_iterations`` HIO steps and return the reconstructed object."""
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive.")

        for _ in range(n_iterations):
            self.step()

        return self.object_field
