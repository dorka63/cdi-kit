"""Hybrid Input-Output engine for single-frame far-field CDI."""

from __future__ import annotations

import numpy as np

from ptypy.custom.cdi_common import CDIProjectionEngine
from ptypy.engines import register


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


@register()
class CDIHIO(CDIProjectionEngine):
    """
    Hybrid Input-Output for single-frame far-field CDI.

    Defaults:

    [name]
    default = CDIHIO
    type = str
    help =

    [beta]
    default = 0.9
    type = float
    lowlim = 0.0
    uplim = 1.0
    help = HIO feedback parameter

    """

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        ptycho_parent.citations.add_article(**HIO_ARTICLE)

    def object_update(self, current, candidate, support):
        return np.where(support, candidate, current - self.p.beta * candidate)