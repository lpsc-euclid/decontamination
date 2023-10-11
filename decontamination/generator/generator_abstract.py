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
        HEALPix pixel indices of the footprint.
    nest : bool
        Tf True, assumes NESTED pixel ordering, otherwise, RING pixel ordering (default: **True**).
    """

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, nest: bool = True):

        ################################################################################################################

        self._nside = nside

        ################################################################################################################

        theta, phi = hp.pix2ang(nside, footprint, nest = nest)

        x_diamonds, y_diamonds = thetaphi2xy(theta, phi)

        self._x_diamonds = x_diamonds
        self._y_diamonds = y_diamonds

########################################################################################################################
