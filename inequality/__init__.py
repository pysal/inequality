"""
:mod:`inequality` --- Spatial Inequality Analysis
=================================================

"""

from . import _version, gini, theil
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
    simpson_sd,
    simpson_so,
    theil_th,
)

__version__ = _version.get_versions()["version"]
