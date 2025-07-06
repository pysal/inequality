"""
:mod:`inequality` --- Spatial Inequality Analysis
=================================================

"""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import atkinson, gini, schutz, theil, wolfson

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("inequality")
