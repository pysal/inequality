"""
:mod:`inequality` --- Spatial Inequality Analysis
=================================================

"""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import atkinson, gini, schutz, theil, wolfson
from ._indices import (
    abundance,
    ellison_glaeser_egg,
    ellison_glaeser_egg_pop,
    fractionalization_gs,
    gini_gi,
    gini_gi_m,
    gini_gig,
    herfindahl_hd,
    hoover_hi,
    isolation_ii,
    isolation_isg,
    margalev_md,
    maurel_sedillot_msg,
    maurel_sedillot_msg_pop,
    menhinick_mi,
    modified_segregation_msg,
    polarization,
    segregation_gsg,
    shannon_se,
    similarity_w_wd,
    simpson_sd,
    simpson_so,
    theil_th,
)

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("inequality")
