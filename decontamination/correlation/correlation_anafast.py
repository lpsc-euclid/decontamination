# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import healpy as hp

from scipy.special import legendre

from . import correlation_abstract

########################################################################################################################

# noinspection PyPep8Naming
class Correlation_Anafast(correlation_abstract.Correlation_Abstract):

    """
    ??? angular correlation function.

    Parameters
    ----------
    catalog_lon : np.ndarray
        Galaxy catalog longitudes (in degrees).
    catalog_lat : np.ndarray
        Galaxy catalog latitudes (in degrees).
    nside : int
        The HEALPix nside parameter.
    min_sep : float
        Minimum separation being considered (in arcmin).
    max_sep : float
        Maximum separation being considered (in arcmin).
    n_bins : int
        Number of angular bins.
    """

    ####################################################################################################################

    def __init__(self, catalog_lon: np.ndarray, catalog_lat: np.ndarray, nside: int, min_sep: float, max_sep: float, n_bins: int):

        ################################################################################################################

        super().__init__(min_sep, max_sep, n_bins)

        ################################################################################################################

        self._nside = nside

        ################################################################################################################

        self._theta = np.linspace(
            np.radians(min_sep / 60.0),
            np.radians(max_sep / 60.0),
            n_bins
        )

        ################################################################################################################

        self._pixels = hp.ang2pix(nside, catalog_lon, catalog_lat, nest = False, lonlat = True)

    ####################################################################################################################

    @property
    def nside(self):

        return self._nside

    ####################################################################################################################

    def _cell2corr(self, cell):

        ell = np.arange(cell.shape[0])

        pl = np.array([legendre(n)(np.cos(self._theta)) for n in ell])

        return np.sum((2.0 * ell + 1.0) * np.outer(cell, pl), axis = 0) / (4.0 * np.pi)

    ####################################################################################################################

    def calculate(self) -> typing.Tuple[np.ndarray, np.ndarray]:

        """
        Calculates the angular correlation function.

        .. math::
            \\xi(\\theta)=\\frac{1}{\sqrt{4\\pi}}\\sum_{l=0}^{2\\times\\text{nside}}(2l+1)\\,C_l\\,P_l(\\cos\\theta)

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            The bin of angles :math:`\\theta` and the angular correlations :math:`\\xi(\\theta)`.
        """

        ################################################################################################################

        cell = hp.anafast(self._pixels, lmax = 2 * self._nside)

        ################################################################################################################

        return self._theta, self._cell2corr(cell)

########################################################################################################################
