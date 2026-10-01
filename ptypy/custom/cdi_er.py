"""Error Reduction engine for single-frame far-field CDI."""

from __future__ import annotations

import numpy as np

from ptypy.core.manager import Full, Vanilla
from ptypy.custom.cdi_common import (
    cdi_support,
    masked_amplitude_error,
    masked_modulus_projection,
)
from ptypy.engines import register
from ptypy.engines.base import BaseEngine


ER_ARTICLE = dict(
    comment="The Error Reduction phase-retrieval algorithm",
    title="A phase retrieval algorithm for real and imaginary objects",
    author="Gerchberg R. W. and Saxton W. O.",
    journal="Optik",
    volume=35,
    year=1972,
    page=237,
)


@register()
class CDIER(BaseEngine):
    """
    Error Reduction for single-frame far-field CDI.

    Defaults:

    [name]
    default = CDIER
    type = str
    help =

    """

    SUPPORTED_MODELS = [Vanilla, Full]

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        ptycho_parent.citations.add_article(**ER_ARTICLE)
        self._amplitudes = {}

    def engine_initialize(self):
        for storage in self.pr.storages.values():
            storage.fill(1.0)

    def engine_prepare(self):
        self._amplitudes = {
            pod_id: np.sqrt(pod.diff) for pod_id, pod in self.pods.items()
        }

    def engine_iterate(self, num=1):
        error_dct = {}

        for _ in range(num):
            for name, diff_view in self.di.views.items():
                if not diff_view.active:
                    continue

                for pod_id, pod in diff_view.pods.items():
                    current = pod.object.copy()
                    amplitude = self._amplitudes[pod_id]
                    detector_field = pod.fw(pod.probe * current)

                    amplitude_error = masked_amplitude_error(
                        detector_field, amplitude, pod.mask
                    )
                    projected = masked_modulus_projection(
                        detector_field, amplitude, pod.mask
                    )

                    updated = pod.bw(projected) * cdi_support(self.ptycho, pod)
                    pod.object = updated

                    norm = np.linalg.norm(current)
                    change = np.linalg.norm(updated - current) / norm if norm > 0 else 0.0

                    error_dct[name] = np.array([amplitude_error, 0.0, change])

            self.curiter += 1

        return error_dct

    def engine_finalize(self):
        pass