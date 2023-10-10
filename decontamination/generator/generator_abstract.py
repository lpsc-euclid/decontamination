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
    """

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, nest: bool = True):

        """
        ???

        Parameters
        ----------
        nside : int
            ???
        footprint : np.ndarray
            ???
        nest : bool
            ???
        """

        ################################################################################################################

        self._nside = nside

        ################################################################################################################

        theta, phi = hp.pix2ang(nside, footprint, nest = nest)

        x_diamonds, y_diamonds = thetaphi2xy(theta, phi)

        self._x_diamonds = x_diamonds
        self._y_diamonds = y_diamonds

########################################################################################################################
