"""
:mod:`inequality` --- Spatial Inequality Analysis
=================================================

"""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import atkinson, gini, pen, polarization, schutz, theil, wolfson
from .atkinson import Atkinson
from .gini import Gini, Gini_Spatial
from .polarization import S
from .schutz import Schutz
from .theil import Theil, TheilD, TheilDSim

__all__ = [
    # submodules
    "atkinson",
    "gini",
    "pen",
    "polarization",
    "schutz",
    "theil",
    "wolfson",
    # measure classes (GL#38)
    "Atkinson",
    "Gini",
    "Gini_Spatial",
    "S",
    "Schutz",
    "Theil",
    "TheilD",
    "TheilDSim",
]

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("inequality")
