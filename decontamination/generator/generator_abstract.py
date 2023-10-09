# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np

import healpy as hp

from . import thetaphi2xy

########################################################################################################################

# noinspection PyPep8Naming
class Generator_abstract(object):

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, nest: bool = True):

        """
        Parameters
        ----------
        nside : int
            ???
        footprint : np.ndarray
            ???
        nest : bool
            ???
        """

        theta, phi = hp.pix2ang(nside, footprint, nest = nest)

        x_center, y_center = thetaphi2xy(theta, phi)

        self._x_center = x_center
        self._y_center = y_center

########################################################################################################################
