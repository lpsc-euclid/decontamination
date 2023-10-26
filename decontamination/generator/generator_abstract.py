# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np

import healpy as hp

from . import thetaphi2xy

########################################################################################################################

# noinspection PyPep8Naming
class Generator_Abstract(object):

    """
    Abstract galaxy catalog generator.

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    footprint : np.ndarray
        HEALPix indices of the region where galaxies are generated.
    nest : bool
        If **True**, assumes NESTED pixel ordering, otherwise, RING pixel ordering (default: **True**).
    lonlat : bool
        If **True**, assumes ??? (default: **True**).
    """

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, nest: bool = True, lonlat: bool = True):

        self._nside = nside

        self._footprint = footprint

        self._nest = nest
        self._lonlat = lonlat

########################################################################################################################
