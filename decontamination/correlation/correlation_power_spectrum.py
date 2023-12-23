# -*- coding: utf-8 -*-
########################################################################################################################

import math
import typing

import numpy as np
import healpy as hp

from scipy.special import legendre

from . import correlation_abstract

########################################################################################################################

try:

    import xpol

except ImportError:

    xpol = None

########################################################################################################################

# noinspection PyPep8Naming
class Correlation_PowerSpectrum(correlation_abstract.Correlation_Abstract):

    """
    Angular correlation function from power spectrum.

    Parameters
    ----------
    footprint : np.ndarray
        HEALPix indices of the region where correlation must be calculated.
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

    def __init__(self, footprint: np.ndarray, catalog_lon: np.ndarray, catalog_lat: np.ndarray, nside: int, min_sep: float, max_sep: float, n_bins: int):

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
        # BUILD THE GALAXY NUMBER DENSITY MAP                                                                          #
        ################################################################################################################

        galaxy_pixels = hp.ang2pix(nside, catalog_lon, catalog_lat, nest = False, lonlat = True)

        ################################################################################################################

        self._full_sky_number_density = np.zeros(hp.nside2npix(nside), dtype = np.float32)

        np.add.at(self._full_sky_number_density, galaxy_pixels, 1.0)

        ################################################################################################################
        # BUILD THE FOOTPRINT                                                                                          #
        ################################################################################################################

        self._full_sky_footprint = np.zeros(hp.nside2npix(nside), dtype = np.float32)

        self._full_sky_footprint[footprint] = 1.0

    ####################################################################################################################

    @property
    def nside(self):

        return self._nside

    ####################################################################################################################

    def _cl2correlation(self, ell, cell):

        pl = np.array([legendre(n)(np.cos(self._theta)) for n in ell])

        return np.sum((2.0 * ell + 1.0) * np.outer(cell, pl), axis = 0) / math.sqrt(4.0 * np.pi)

    ####################################################################################################################

    def calculate(self, library: str) -> typing.Tuple[np.ndarray, np.ndarray]:

        """
        Calculates the angular correlation function.

        .. math::
            \\xi(\\theta)=\\frac{1}{\\sqrt{4\\pi}}\\sum_{l=0}^{2\\times\\text{nside}}(2l+1)\\,C_l\\,P_l(\\cos\\theta)

        Parameters
        ----------
        library : str
            Library to be used for calculating the power spectrum ("xpol", "healpy").

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            The bin of angles :math:`\\theta` and the angular correlations :math:`\\xi(\\theta)`.
        """

        ################################################################################################################
        # LIBRARY = XPOL                                                                                               #
        ################################################################################################################

        if library == 'xpol':

            if xpol is None:

                raise ImportError('Xpol is not installed.')

            ####

            zeros = np.zeros_like(self._full_sky_number_density)

            binning = xpol.Bins.fromdeltal(2, 2 * self._nside, 1)

            xp = xpol.Xpol(self._full_sky_footprint, bins = binning, verbose = False)

            pcl, _ = xp.get_spectra(np.asarray([
                [self._full_sky_number_density], # T
                [zeros                        ], # E
                [zeros                        ], # B
            ]), remove_dipole = False)

            cell = pcl[0, :].flatten().astype(np.float64)

            ell = binning.lbin.astype(np.int64)

        ################################################################################################################
        # LIBRARY = ANAFAST                                                                                            #
        ################################################################################################################

        else:

            ma = hp.ma(self._full_sky_number_density)

            ma.mask = np.logical_not(self._full_sky_footprint)

            cell = hp.anafast(ma.filled(), lmax = 2 * self._nside)

            ell = np.arange(cell.shape[0])

        ################################################################################################################

        return self._theta, self._cl2correlation(ell, cell)

########################################################################################################################
