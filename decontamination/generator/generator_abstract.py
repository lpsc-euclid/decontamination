# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np

import healpy as hp

from . import thetaphi2xy

########################################################################################################################

# noinspection PyPep8Naming
class Generator_Abstract(object):

    """
    Abstract galaxy generator.
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

        diamond_x_center, diamond_y_center = thetaphi2xy(theta, phi)

        self._x_center_diamond = diamond_x_center
        self._y_center_diamond = diamond_y_center

########################################################################################################################
