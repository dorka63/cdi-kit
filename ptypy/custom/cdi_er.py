"""Error Reduction engine for single-frame far-field CDI."""

from __future__ import annotations

from ptypy.custom.cdi_common import CDIProjectionEngine
from ptypy.engines import register


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
class CDIER(CDIProjectionEngine):
    """
    Error Reduction for single-frame far-field CDI.

    Defaults:

    [name]
    default = CDIER
    type = str
    help =

    """

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        ptycho_parent.citations.add_article(**ER_ARTICLE)

    def object_update(self, current, candidate, support):
        return candidate * support