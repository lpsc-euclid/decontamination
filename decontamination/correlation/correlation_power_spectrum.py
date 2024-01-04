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
        Minimum separation being considered (in arcmins).
    max_sep : float
        Maximum separation being considered (in arcmins).
    n_bins : int
        Number of angular bins.
    """

    ####################################################################################################################

    def __init__(self, footprint: np.ndarray, catalog_lon: np.ndarray, catalog_lat: np.ndarray, nside: int, min_sep: float, max_sep: float, n_bins: int, library: str = 'xpol'):

        ################################################################################################################

        super().__init__(min_sep, max_sep, n_bins)

        ################################################################################################################

        self._nside = nside

        self._library = library

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

        self._full_sky_contrast = np.zeros(hp.nside2npix(nside), dtype = np.float32)

        np.add.at(self._full_sky_contrast, galaxy_pixels, 1.0)

        ################################################################################################################
        # BUILD THE FOOTPRINT                                                                                          #
        ################################################################################################################

        self._full_sky_footprint = np.zeros(hp.nside2npix(nside), dtype = np.float32)

        self._full_sky_footprint[footprint] = 1.0

        ################################################################################################################
        # BUILD THE CONTRAST                                                                                           #
        ################################################################################################################

        mean = np.mean(self._full_sky_contrast[footprint])

        self._full_sky_contrast[footprint] = (self._full_sky_contrast[footprint] - mean) / mean

    ####################################################################################################################

    @property
    def nside(self):

        return self._nside

    ####################################################################################################################

    def _cell2correlation(self, ell, cell):

        pl_cos_theta = np.array([legendre(l)(np.cos(self._theta)) for l in ell]).T

        return np.sum((2.0 * ell + 1.0) * cell * pl_cos_theta, axis = 1) / math.sqrt(4.0 * np.pi)

    ####################################################################################################################

    def calculate(self, estimator: str, random_contrast: typing.Optional[np.ndarray] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        pass

    ####################################################################################################################

    def calculate_xy(self, contrast1: np.ndarray, contrast2: np.ndarray, library: str) -> typing.Tuple[np.ndarray, np.ndarray]:

        """
        Calculates the angular correlation function.

        .. math::
            \\xi(\\theta)=\\frac{1}{\\sqrt{4\\pi}}\\sum_{l=0}^{2\\times\\text{nside}}(2l+1)\\,C_l\\,P_l(\\cos\\theta)

        Parameters
        ----------
        library : str
            Library to be used for calculating the :math:`\\text{pseudo}-C_l` inside the footprint ("xpol", "healpy").

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            The bin of angles :math:`\\theta` (in arcmins) and the angular correlations :math:`\\xi(\\theta)`.
        """

        ################################################################################################################
        # LIBRARY = XPOL                                                                                               #
        ################################################################################################################

        if self._library == 'xpol':

            if xpol is None:

                raise ImportError('Xpol is not installed.')

            ####

            zeros = np.zeros_like(contrast1)

            binning = xpol.Bins.fromdeltal(0, 2 * self._nside, 1)

            xp = xpol.Xpol(self._full_sky_footprint, bins = binning, verbose = False)

            pcl, _ = xp.get_spectra(m1 = np.asarray([
                [contrast1], # T
                [zeros],     # E
                [zeros],     # B
            ]), m2 = np.asarray([
                [contrast2], # T
                [zeros],     # E
                [zeros],     # B
            ]) if contrast2 is not None else None, pixwin = True, remove_dipole = False)

            cell = pcl[0].astype(np.float64)

        ################################################################################################################
        # LIBRARY = ANAFAST                                                                                            #
        ################################################################################################################

        else:

            ma = hp.ma(self._full_sky_contrast)

            ma.mask = np.logical_not(self._full_sky_footprint)

            cell = hp.anafast(ma.filled(), lmax = 2 * self._nside)

        ################################################################################################################

        ell = np.arange(cell.shape[0], dtype = np.int64)

        return 60.0 * np.degrees(self._theta), self._cell2correlation(ell, cell)

########################################################################################################################
