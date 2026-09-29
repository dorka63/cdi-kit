"""CPU Error-Reduction reconstruction for coherent diffraction imaging."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cdi_common import CDIProblem


Array = np.ndarray


@dataclass
class ErrorReduction:
    """Classical CDI Error-Reduction reconstruction.

    The update rule is

    .. math::

        \\rho_{k+1} = P_S P_M(\\rho_k),

    where ``P_M`` is the masked detector modulus projection and ``P_S`` is
    the real-space support projection.
    """

    problem: CDIProblem
    object_field: Array
    amplitude_errors: list[float] = field(default_factory=list)
    object_changes: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
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

    def step(self) -> float:
        """Perform one Error-Reduction iteration and return its data error."""
        current = self.object_field

        candidate, amplitude_error = self.problem.data_projection(current)

        updated = candidate * self.problem.support

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
        """Run ``n_iterations`` ER steps and return the reconstructed object."""
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive.")

        for _ in range(n_iterations):
            self.step()

        return self.object_field