"""RAAR engine for support-constrained CDI reconstruction."""

from __future__ import annotations

import numpy as np

from ptypy.engines import register
from ptypy.custom.cdi_common import CDIProjectionEngine


@register()
class CDIRAAR(CDIProjectionEngine):
    """Relaxed averaged alternating reflections for CDI.

    Defaults:

    [name]
    default = CDIRAAR
    type = str
    help = Name of the CDI RAAR engine

    [beta]
    default = 0.9
    type = float
    help = RAAR relaxation parameter in the closed interval [0, 1]
    """

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)

        beta = float(self.p.beta)

        if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError(
                "CDIRAAR beta must be finite and in [0, 1]."
            )

        self.p.beta = beta

        self.ptycho.citations.add_article(
            title=(
                "Relaxed averaged alternating reflections "
                "for diffraction imaging"
            ),
            author="Luke D. R.",
            journal="Inverse Problems",
            volume=21,
            year=2005,
            page="37–50",
            doi="10.1088/0266-5611/21/1/004",
            comment="The RAAR phase-retrieval algorithm",
        )

    def object_update(self, current, candidate, support):
        """Combine the current field and the modulus-projected field."""
        beta = self.p.beta

        return np.where(
            support,
            candidate,
            beta * current + (1.0 - 2.0 * beta) * candidate,
        )