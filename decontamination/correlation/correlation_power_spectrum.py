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

    .. math::
        \\xi(\\theta)=\\frac{1}{\\sqrt{4\\pi}}\\sum_{l=0}^{2\\times\\text{nside}}(2l+1)\\,C_l\\,P_l(\\cos\\theta)

    Parameters
    ----------
    catalog_lon : np.ndarray
        Galaxy catalog longitudes (in degrees).
    catalog_lat : np.ndarray
        Galaxy catalog latitudes (in degrees).
    footprint : np.ndarray
        HEALPix indices of the region where correlation must be calculated.
    nside : int
        The HEALPix nside parameter.
    min_sep : float
        Minimum galaxy separation being considered (in arcmins).
    max_sep : float
        Maximum galaxy separation being considered (in arcmins).
    n_bins : int
        Number of angular bins.
    library : str
        Library to be used for calculating the :math:`\\text{pseudo-}C_l` inside the footprint (“xpol”, “healpy”).
    """

    ####################################################################################################################

    def __init__(self, catalog_lon: np.ndarray, catalog_lat: np.ndarray, footprint: np.ndarray, nside: int, min_sep: float, max_sep: float, n_bins: int, library: str = 'xpol'):

        ################################################################################################################

        super().__init__(min_sep, max_sep, n_bins)

        ################################################################################################################

        self._footprint = footprint

        self._nside = nside

        self._library = library

        ################################################################################################################

        self._theta = np.linspace(
            np.radians(min_sep / 60.0),
            np.radians(max_sep / 60.0),
            n_bins
        )

        ################################################################################################################
        # BUILD THE FOOTPRINT                                                                                          #
        ################################################################################################################

        self._full_sky_footprint = np.zeros(hp.nside2npix(nside), dtype = np.float32)

        self._full_sky_footprint[footprint] = 1.0

        ################################################################################################################
        # BUILD THE CONTRAST                                                                                           #
        ################################################################################################################

        self._data_contrast = self._build_full_sky_contrast(
            catalog_lon,
            catalog_lat
        )

        ################################################################################################################
        # CORRELATE IT                                                                                                 #
        ################################################################################################################

        self._dd = self._correlate(self._data_contrast, None)

    ####################################################################################################################

    @property
    def footprint(self):

        return self._footprint

    ####################################################################################################################

    @property
    def nside(self):

        return self._nside

    ####################################################################################################################

    @property
    def library(self):

        return self._library

    ####################################################################################################################

    def _build_full_sky_contrast(self, catalog_lon: np.ndarray, catalog_lat: np.ndarray) -> np.ndarray:

        ################################################################################################################

        galaxy_pixels = hp.ang2pix(self._nside, catalog_lon, catalog_lat, nest = False, lonlat = True)

        result = np.zeros(hp.nside2npix(self._nside), dtype = np.float32)

        np.add.at(result, galaxy_pixels, 1.0)

        ################################################################################################################

        mean = np.mean(result[self._footprint])

        result[self._footprint] = (result[self._footprint] - mean) / mean

        ################################################################################################################

        return result

    ####################################################################################################################

    def _cell2correlation(self, ell, cell):

        pl_cos_theta = np.array([legendre(l)(np.cos(self._theta)) for l in ell]).T

        return np.sum((2.0 * ell + 1.0) * cell * pl_cos_theta, axis = 1) / math.sqrt(4.0 * np.pi)

    ####################################################################################################################

    def _correlate(self, contrast1: typing.Optional[np.ndarray], contrast2: typing.Optional[np.ndarray]) -> typing.Tuple[np.ndarray, np.ndarray]:

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

            pcl, _ = xp.get_spectra(
                m1 = np.asarray([
                    [contrast1], # T
                    [zeros],     # E
                    [zeros],     # B
                ]),
                m2 = np.asarray([
                    [contrast2], # T
                    [zeros],     # E
                    [zeros],     # B
                ]) if contrast2 is not None else None, pixwin = True, remove_dipole = False
            )

            cell = pcl[0].astype(np.float64)

        ################################################################################################################
        # LIBRARY = ANAFAST                                                                                            #
        ################################################################################################################

        else:

            ma1 = hp.ma(contrast1)
            ma1.mask = np.logical_not(self._full_sky_footprint)

            if contrast2 is not None:

                ma2 = hp.ma(contrast2)
                ma2.mask = np.logical_not(self._full_sky_footprint)

                cell = hp.anafast(map1 = ma1.filled(), map2 = ma2.filled(), lmax = 2 * self._nside)

            else:

                cell = hp.anafast(map1 = ma1.filled(), map2 = None, lmax = 2 * self._nside)

        ################################################################################################################

        ell = np.arange(cell.shape[0], dtype = np.int64)

        ################################################################################################################

        return 60.0 * np.degrees(self._theta), self._cell2correlation(ell, cell)

    ####################################################################################################################

    def calculate(self, estimator: str, random_lon: typing.Optional[np.ndarray] = None, random_lat: typing.Optional[np.ndarray] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """
        Calculates the angular correlation function.

        Peebles & Hauser estimator (1974):

        .. math::
            \\hat{\\xi}=\\frac{DD}{RR}-1

        1st Landy & Szalay estimator (1993):

        .. math::
            \\hat{\\xi}=\\frac{DD-2DR-RR}{RR}

        2nd Landy & Szalay estimator (1993):

        .. math::
            \\hat{\\xi}=\\frac{DD-DR-RD-RR}{RR}

        Parameters
        ----------
        estimator : str
            Estimator being considered ("dd", "rr", "dr", "rd", "peebles_hauser", "landy_szalay_1", "landy_szalay_2").
        random_lon : np.ndarray, default: None
            Random catalog longitudes (in degrees). For Peebles & Hauser and Landy & Szalay estimators only.
        random_lat : np.ndarray, default: None
            Random catalog latitudes (in degrees). For Peebles & Hauser and Landy & Szalay estimators only.

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            The bin of angles :math:`\\theta` (in arcmins), the angular correlations :math:`\\xi(\\theta)`.
        """

        ################################################################################################################

        if estimator == 'dd':

            return self._calculate_xy(self._data_contrast, None)

        ################################################################################################################

        if random_lon is None\
           or                \
           random_lat is None:

            raise ValueError(f'Parameters `random_lon` and `random_lat` have be provided with estimator `{estimator}`.')

        ################################################################################################################

        random_contrast = self._build_full_sky_contrast(
            random_lon,
            random_lat
        )

        ################################################################################################################

        if estimator == 'rr':
            return self._calculate_xy(random_contrast, None)
        if estimator == 'dr':
            return self._calculate_xy(self._data_contrast, random_contrast)
        if estimator == 'rd':
            return self._calculate_xy(random_contrast, self._data_contrast)
        if estimator == 'peebles_hauser':
            return self._calculate_xi(random_contrast, False, False)
        if estimator == 'landy_szalay_1':
            return self._calculate_xi(random_contrast, True, False)
        if estimator == 'landy_szalay_2':
            return self._calculate_xi(random_contrast, True, True)

        ################################################################################################################

        raise ValueError('Invalid estimator (`dd`, `rr`, `dr`, `rd`, `peebles_hauser`, `landy_szalay1`, `landy_szalay_2`)')

    ####################################################################################################################

    def _calculate_xy(self, contrast1: typing.Optional[np.ndarray], contrast2: typing.Optional[np.ndarray]) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if contrast1 is self._data_contrast or contrast2 is not None:

            xy = self._correlate(contrast1, contrast2)

        else:

            xy = self._dd

        ################################################################################################################

        return xy[0], xy[1], np.zeros_like(xy[1])

    ####################################################################################################################

    def _calculate_xi(self, random_contrast: np.ndarray, with_dr: bool, with_rd: bool) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        rr = self._calculate_xy(random_contrast, None)

        if with_dr:

            dr = self._calculate_xy(self._data_contrast, random_contrast)

            if with_rd:

                rd = self._calculate_xy(random_contrast, self._data_contrast)

                return self._dd[0], (self._dd[1] - dr[1] - rd[1] + rr[1]) / rr[1], self._dd[2]

            else:

                return self._dd[0], (self._dd[1] - 2.0 * dr[1] + rr[1]) / rr[1], self._dd[2]
        else:

            return self._dd[0], self._dd[1] / rr[1] - 1.0, self._dd[2]

########################################################################################################################
